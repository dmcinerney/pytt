import torch
import torch.distributed as dist
from pytt.logger import logger
from pytt.utils import indent


# TODO: add comments
class IterationInfo:
    def __init__(self):
        self.iterator_info = None
        self.train_info = None
        self.val_info = None

    def set_iterator_info(self, iterator_info):
        self.iterator_info = iterator_info

    def set_train_info(self, batch_info):
        self.train_info = batch_info

    def check_initialized(self):
        return self.iterator_info is not None and self.train_info is not None

    def set_val_info(self, batch_info):
        self.val_info = batch_info

    def __add__(self, iteration_info):
        if not (self.check_initialized()
                and iteration_info.check_initialized()):
            raise Exception
        new_iteration_info = self.__class__()

        # set iteration_info
        new_iteration_info.set_iterator_info(iteration_info.iterator_info)

        # set train_info
        new_iteration_info.train_info = self.train_info\
                                        +iteration_info.train_info

        # set val_info
        if iteration_info.val_info is not None:
            new_iteration_info.val_info = iteration_info.val_info
            if self.val_info is not None:
                new_iteration_info.val_info += self.val_info

        return new_iteration_info

    def __radd__(self, i):
        if i != 0:
            raise Exception
        return self

    def __str__(self):
        step_info = str(self.iterator_info)
        step_info += "\n  TRAIN\n"+indent(str(self.train_info), "    ")
        if self.val_info is not None:
            step_info += "\n  VAL\n"+indent(str(self.val_info), "    ")
        return step_info

    def to_tensor(self):
        tensors = []
        tensors.append(self.iterator_info.to_tensor().float())
        tensors.append(self.train_info.to_tensor())
        if self.val_info is not None:
            tensors.append(self.val_info.to_tensor())
        return torch.cat(tensors, 0)

    def from_tensor(self, tensor, isiter=False):
        if not isiter:
            tensor_iter = iter(tensor)
        else:
            tensor_iter = tensor
        self.iterator_info.from_tensor(tensor_iter, isiter=True)
        self.train_info.from_tensor(tensor_iter, isiter=True)
        if self.val_info is not None:
            self.val_info.from_tensor(tensor_iter, isiter=True)
        return self


class BatchInfo:
    def __init__(self, batch_info_dict):
        self.batch_info_dict = batch_info_dict

    def __add__(self, batch_info):
        new_batch_info_dict = {}
        for k in set(self.batch_info_dict.keys()).union(
                     batch_info.batch_info_dict.keys()):
            new_batch_info_dict[k] = self.batch_info_dict[k]\
                                     + batch_info.batch_info_dict[k]
        return self.__class__(new_batch_info_dict)

    def __radd__(self, i):
        if i != 0:
            raise Exception
        return self

    def __str__(self):
        step_info = ""
        first = True
        for (k,v) in sorted(self.batch_info_dict.items(), key=lambda kv: kv[0]):
            if k.startswith('_'):
                continue
            if not first:
                step_info += ", "
            first = False
            step_info += ("%s per instance: " % k)\
                         +str(v/self.batch_info_dict['_batch_length'])
        return step_info

    def to_tensor(self):
        list_of_floats = []
        for k,v in sorted(self.batch_info_dict.items(),
                          key=lambda kv: kv[0]):
            list_of_floats.append(v)
        return torch.tensor(list_of_floats)

    def from_tensor(self, tensor, isiter=False):
        if not isiter:
            tensor_iter = iter(tensor)
        else:
            tensor_iter = tensor
        for k,v in sorted(self.batch_info_dict.items(),
                          key=lambda kv: kv[0]):
            self.batch_info_dict[k] = float(next(tensor_iter).item())
        return self
