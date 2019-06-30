import os
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytt.utils import set_random_state


def setup(rank, world_size, random_state=None, environment_name='MASTER',
          address='localhost', port=12355, backend='nccl'):
    """
    Sets up the distributed process
    """
    # set up environment
    os.environ[environment_name+'_ADDR'] = address
    os.environ[environment_name+'_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # set random state
    if random_state is not None:
        set_random_state(random_state)

def cleanup():
    """
    Cleans up the distributed process
    """
    dist.destroy_process_group()

def _process_func(rank, setup_args, setup_kwargs, func, func_args, func_kwargs):
    """
    Sandwiches function call between setup and cleanup with a local process rank
    of rank

    IMPORTANT NOTE: this should only be used by the distributed_wrapper function
    """
    setup(rank, *setup_args, **setup_kwargs)
    func(*func_args, **func_kwargs)
    cleanup()

def distributed_wrapper(func, nprocs, random_state=None,
                        environment_name='MASTER', address='localhost',
                        port='12355', backend='gloo'):
    """
    Wraps a function, returning a function that spawns multiple processes,
    optionally starting from the given random state

    IMPORTANT NOTE: the given function must take an optional rank_worldsize
    parameter, describing the rank and worldsize
    """
    # create setup_args/kwargs
    setup_args = (nprocs,)
    setup_kwargs = dict(
        random_state=random_state,
        environment_name=environment_name,
        address=address,
        port=port,
        backend=backend,
    )

    # create wrapper function
    def func_wrapper(*func_args, **func_kwargs):
        """
        Spawns multiple processes that call the function with the given args
        """
        mp.spawn(_process_func,
             args=(setup_args, setup_kwargs, func, func_args, func_kwargs),
             nprocs=nprocs,
             join=True)
    return func_wrapper

def collect_tensors_on_rank0(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank != 0:
        dist.send(tensor, 0)
        return None
    else:
        tensors = [tensor]
        for i in range(1,world_size):
            new_tensor = torch.zeros_like(tensor)
            dist.recv(new_tensor, i)
            tensors.append(new_tensor)
        return tensors

def collect_obj_on_rank0(obj):
    tensors = collect_tensors_on_rank0(obj.to_tensor())
    if tensors is None:
        return None
    collected = [copy.deepcopy(obj).from_tensor(tensor) for tensor in tensors]
    return collected
