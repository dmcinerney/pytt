import numpy as np
import torch
import pickle as pkl
import math


def seed_state(seed=0):
    """
    Sets the seed for numpy and torch and makes cuda deterministic as well
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_random_state(filename):
    """
    Saves the random states of both numpy and torch into a file via pickle
    """
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    with open(filename, 'wb') as f:
        pkl.dump(tuple(numpy_state, torch_state), filename)

def load_random_state(filename):
    """
    Loads the random states of both numpy and torch into a file via pickle
    """
    with open(filename, 'rb') as f:
        numpy_state, torch_state = pkl.load(filename)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)

def split(n, k):
    """
    Returns a generator for the lengths of sections if you split an object
    of length n into k approximately equal length parts
    """
    new_size_floor = math.floor(n/k)
    additional = n % k
    for i in range(k):
        if i < additional:
            yield new_size_floor + 1
        else:
            yield new_size_floor