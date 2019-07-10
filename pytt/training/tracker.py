import torch.distributed as dist
from pytt.utils import read_pickle, write_pickle
from pytt.distributed import collect_obj_on_rank0

class Tracker:
    @classmethod
    def load(cls, filename):
        return cls(read_pickle(filename))

    def __init__(self, history=[]):
        self.history = history

    def register_iteration(self, iteration_info):
        self.history.append(iteration_info)
        if dist.is_initialized():
            collected = collect_obj_on_rank0(
                self.history[-1],
                ranks=self.history[-1].iterator_info.subbatches.get_ranks())
            if collected is not None:
                self.history[-1] = sum(collected)
            else:
                self.history = []

    def __str__(self):
        return str(self.history[-1])

    def save(self, filename):
        write_pickle(self.history, filename)
