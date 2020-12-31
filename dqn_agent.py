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
		self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)
		
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
		self.memory.add(state, action, reward, next_state, done)

		self.t_step += 1
		if self.t_step % UPDATE_EVERY == 0:
			if len(self.memory) > BATCH_SIZE:
				self.learn()

	def learn(self):
		states, actions, rewards, next_states, dones = self.memory.sample()

		# Double DQN
		# 1. Action selection
		q_local_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

		# 2. Action evaluation
		q_targets_next = self.qnetwork_target(next_states).gather(1, q_local_actions)
		q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))

		q_expected = self.qnetwork_local(states).gather(1, actions)

		self.optimizer.zero_grad()
		loss = F.mse_loss(q_targets, q_expected)
		loss.backward()
		self.optimizer.step()

		self.update_target_network()

	def update_target_network(self):
		for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
			target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
