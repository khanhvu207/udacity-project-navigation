import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from hyperparams import *
from memory import *
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
	def __init__(self, state_size, action_size, seed):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)

		# Q-networks
		self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
		self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
		
		# Optimizer
		self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

		# (Prioritized) Replay memory
		# self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
		self.memory = Memory(BUFFER_SIZE)
		
		# Update and synchronize Q-target with Q-network every UPDATE_EVERY steps
		self.t_step = 0

	def act(self, state, eps):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.qnetwork_local.eval()
		with torch.no_grad():
			action_values = self.qnetwork_local(state)
		self.qnetwork_local.train()

		# Epsilon-greedy
		if random.random() > eps:
			return np.argmax(action_values.to('cpu').data.numpy())
		else:
			return random.choice(np.arange(self.action_size))

	def step(self, state, action, reward, next_state, done):
		self.add_to_mem(state, action, reward, next_state, done)

		self.t_step += 1
		if self.t_step % UPDATE_EVERY == 0:
			if self.t_step + 5 > BATCH_SIZE:
				self.learn()

	def learn(self):
		mini_batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
		mini_batch = np.array(mini_batch, dtype=object).transpose()
		
		states = np.vstack(mini_batch[0])
		actions = np.vstack(list(mini_batch[1]))
		rewards = np.vstack(list(mini_batch[2]))
		next_states = np.vstack(mini_batch[3])
		dones = np.vstack(mini_batch[4].astype(np.uint8))

		states = torch.from_numpy(states).float().to(device)
		actions = torch.from_numpy(actions).long().to(device)
		rewards = torch.from_numpy(rewards).float().to(device)
		next_states = torch.from_numpy(next_states).float().to(device)
		dones = torch.from_numpy(dones).float().to(device)

		# Double DQN
		# 1. Action selection
		q_local_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

		# 2. Action evaluation
		q_targets_next = self.qnetwork_target(next_states).gather(1, q_local_actions)
		q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))

		q_expected = self.qnetwork_local(states).gather(1, actions)

		errors = torch.abs(q_expected - q_targets).to('cpu').data.numpy()

		# update priority
		for i in range(BATCH_SIZE):
			idx = idxs[i]
			p = self.memory._get_priority(errors[i])
			assert(p != 0)
			self.memory.update(idx, p)

		self.optimizer.zero_grad()
		
		# Loss is scaled with Importance Sampling weight
		loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(q_targets, q_expected)).mean()
		loss.backward()
		
		self.optimizer.step()
		self.update_target_network()

	def update_target_network(self):
		for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
			target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

	def add_to_mem(self, state, action, reward, next_state, done):
		self.qnetwork_local.eval()
		self.qnetwork_target.eval()
		
		with torch.no_grad():
			q_local = self.qnetwork_local(torch.FloatTensor(state).to(device)).to('cpu').data.numpy()[action]
			# Double DQN
			q_local_action = np.argmax(self.qnetwork_local(torch.FloatTensor(next_state).to(device)).to('cpu').data.numpy())
			q_target_next = self.qnetwork_target(torch.FloatTensor(next_state).to(device)).to('cpu').data.numpy()[q_local_action]
		
		self.qnetwork_local.train()
		self.qnetwork_target.train()
		q_target = reward + (GAMMA * q_target_next * (1 - done))
		td_error = abs(q_target - q_local)
		self.memory.add(td_error, (state, action, reward, next_state, done))
