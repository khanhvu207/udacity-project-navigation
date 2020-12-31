import random
import numpy as np
from .SumTree import SumTree

class Memory:
	e = 0.01
	a = 0.2
	beta = 0.4
	beta_increment_per_sampling = 0.001

	def __init__(self, capacity):
		self.tree = SumTree(capacity)
		self.capacity = capacity

	def _get_priority(self, error):
		p = (np.abs(error) + self.e) ** self.a
		return p

	def add(self, error, sample):
		p = self._get_priority(error)
		self.tree.add(p, sample)

	def sample(self, n):
		batch = []
		idxs = []
		segment = self.tree.total() / n
		priorities = []
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
		for i in range(n):
			a = segment * i
			b = segment * (i + 1)
			s = random.uniform(a, b)
			(idx, p, data) = self.tree.get(s)
			if p == 0:
				p += self.e
			priorities.append(p)
			batch.append(data)
			idxs.append(idx)

		sampling_probabilities = priorities / self.tree.total()
		is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
		if (np.isinf(is_weight.max())):
			print(self.tree.n_entries, sampling_probabilities, -self.beta, idxs)
		is_weight /= is_weight.max()

		return batch, idxs, is_weight

	def update(self, idx, error):
		p = self._get_priority(error)
		self.tree.update(idx, p)
