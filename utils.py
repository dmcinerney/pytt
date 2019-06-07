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

def get_random_state():
    """
    Returns the random states of both numpy and torch in a tuple
    """
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random_state = (numpy_state, torch_state)
    return random_state

def set_random_state(random_state):
    """
    Loads the random states of both numpy and torch from a tuple
    """
    numpy_state, torch_state = random_state
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)

def write_pickle(obj, filename):
    """
    Write an object using pickle
    """
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)

def read_pickle(filename):
    """
    Read an object from a pickle file
    """
    with open(filename, 'rb') as f:
        return pkl.load(f)

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

def split_range(n, k, i):
    """
    Returns a the range of the indices of section i if you split an object
    of length n into k approximately equal length parts
    """
    if i >= k:
        raise Exception
    new_size_floor = math.floor(n/k)
    additional = n % k
    if i < additional:
        offset = (new_size_floor + 1)*i
        return offset, offset + new_size_floor + 1
    else:
        offset = (new_size_floor + 1)*additional + new_size_floor*(i-additional)
        return offset, offset + new_size_floor

def nan_to_zero_hook(grad):
    """
    A backward hook function that sets all nan grads to zero
    """
    new_grad = grad.clone()
    if (new_grad != new_grad).any():
        print("Warning: NaN encountered!")
        print(grad)
    new_grad[new_grad != new_grad] = 0
    return new_grad
