import math
import pickle as pkl
import numpy as np
import torch
import torch.nn.functional as F

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

class MultiBatchGradMod:
    def __init__(self, num_instances):
        self.num_instances = num_instances

    def __call__(self, parameters):
        world_size = torch.distributed.get_world_size()\
                     if torch.distributed.is_initialized() else 1
        for p in parameters:
            p.grad = p.grad*world_size/self.num_instances

def get_max_dims(tensors):
    """
    Returns None if the tensors are all the same size and the maximum size in
    each dimension otherwise
    """
    if len(tensors) <= 0:
        return None
    dim = tensors[0].dim()
    max_size = [0]*dim
    different = False
    for tensor in tensors:
        if tensor.dim() != dim:
            raise Exception
        for i in range(dim):
            if not different:
                different = max_size[i] != tensor.size(i)
            max_size[i] = max(max_size[i], tensor.size(i))
    if different:
        return max_size
    else:
        return None

def pad_and_concat(tensors, max_size=None, auto=True):
    """
    Returns concatenated tensors with the added batch dimension being first
    """
    if auto:
        if max_size is not None:
            raise Exception("Must turn auto off to specify max size.")
        max_size = get_max_dims(tensors)
    concatenated_tensor = []
    for i,tensor in enumerate(tensors):
        if i == 0:
            dim = tensor.dim()
        if tensor.dim() != dim:
            raise Exception("Number of dimensions does not match!")
        if max_size is not None:
            padding = []
            for i in range(dim-1,-1,-1):
                diff = max_size[i]-tensor.size(i)
                if diff < 0:
                    raise Exception(
                        "Tensor dim greater than specified max size!")
                padding.extend([0,diff])
            new_tensor = F.pad(tensor, tuple(padding))
        else:
            if i == 0:
                shape = tensor.shape
            if tensor.shape != shape:
                raise Exception(
                    "When auto is turned off and max_size is None, "\
                    + "tensor shapes must match!")
            new_tensor = tensor
        concatenated_tensor.append(new_tensor.view(1,*new_tensor.size()))
    concatenated_tensor = torch.cat(concatenated_tensor, 0)
    return concatenated_tensor

def indent(string, indentation):
    return indentation+string.replace("\n", "\n"+indentation)

class IndexIter:
    def __init__(self, start, end):
        self.set_offset(start)
        self.end = end

    def set_offset(self, offset):
        self.offset = offset

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= self.end:
            raise StopIteration
        self.offset += 1
        return self.offset - 1

    def peek(self):
        return self.offset

