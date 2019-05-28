import numpy as np
import torch
import pickle as pkl


def seed_state(seed=0):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def save_random_state(filename):
	numpy_state = np.random.get_state()
	torch_state = torch.get_rng_state()
	with open(filename, 'wb') as f:
		pkl.dump(tuple(numpy_state, torch_state), filename)

def load_random_state(filename):
	with open(filename, 'rb') as f:
		numpy_state, torch_state = pkl.load(filename)
	np.random.set_state(numpy_state)
	torch.set_rng_state(torch_state)