import copy
import torch.distributed as dist
from pytt.utils import read_pickle, write_pickle
from pytt.distributed import collect_tensors_on_rank0

class History:
    @classmethod
    def load(cls, filename):
        return cls(read_pickle(filename))

    def __init__(self, history=[]):
        self._history = history

    def register_iteration(self, iteration_info):
        if len(self._history) == 0\
           or self._history[-1].iterator_info.take_step:
            self._history.append(iteration_info)
        else:
            self._history[-1] = self._history[-1] + iteration_info

        self._history[-1].log_iteration(full_batch=False)

        if self._history[-1].iterator_info.take_step:
            if dist.is_initialized():
                collected = self.collect_info_on_rank0(self._history[-1])
                if collected is not None:
                    self._history[-1] = sum(collected)
                    self._history[-1].log_iteration()
                else:
                    self._history = []
            else:
                self._history[-1].log_iteration()

    def collect_info_on_rank0(self, iteration_info):
        tensors = collect_tensors_on_rank0(iteration_info.to_tensor())
        if tensors is None:
            return None
        collected = [copy.deepcopy(iteration_info).from_tensor(tensor) for tensor in tensors]
        return collected

    @property
    def history(self):
        if len(self._history) == 0\
           or self._history[-1].iterator_info.take_step:
            return self._history
        else:
            return self._history[:-1]

    def save(self, filename):
        write_pickle(self.history, filename)