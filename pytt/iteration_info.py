import numpy as np
import torch
import torch.distributed as dist
from pytt.logger import logger
from pytt.utils import indent, IndexIter, pad_and_concat


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

    def write_to_tensorboard(self, writers):
        if not self.check_initialized():
            raise Exception
        self.train_info.write_to_tensorboard(
            writers['train'], self.iterator_info)
        if self.val_info is not None:
            self.val_info.write_to_tensorboard(
                writers['val'], self.iterator_info)

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

    def from_tensor(self, tensor, index_iter=None):
        if index_iter is None:
            index_iter = IndexIter(0,tensor.size(0))
        self.iterator_info.from_tensor(tensor, index_iter=index_iter)
        self.train_info.from_tensor(tensor, index_iter=index_iter)
        if self.val_info is not None:
            self.val_info.from_tensor(tensor, index_iter=index_iter)
        return self


class BatchInfo:
    def __init__(self, batch_length, batch=None, batch_outputs=None, batch_stats=None, loss=None):
        self.batch_length = batch_length
        self.batch = batch
        self.batch_outputs = batch_outputs
        self.batch_stats = self.stats()\
                           if batch_stats is None else\
                           batch_stats
        if loss is not None:
            self.batch_stats = {'loss':loss.item(), **self.batch_stats}
        self.filter()

    def stats(self):
        return {}

    def filter(self):
        self.batch_outputs = None
        self.batch = None

    def write_to_tensorboard(self, writer, iterator_info):
        global_step = iterator_info.batches_seen
        for k,v in self.batch_stats.items():
            writer.add_scalar(k, v/self.batch_length, global_step)

    def __add__(self, batch_info):
        new_batch_stats = {}
        for k in set(self.batch_stats.keys()).union(
                     batch_info.batch_stats.keys()):
            new_batch_stats[k] = self.batch_stats[k]\
                                     + batch_info.batch_stats[k]
        new_batch_length = self.batch_length + batch_info.batch_length
        if self.batch_outputs is not None:
            new_batch_outputs = {}
            for k in set(self.batch_outputs.keys()).union(
                         batch_info.batch_outputs.keys()):
                new_batch_outputs[k] = pad_and_concat(
                    [self.batch_outputs[k],
                     batch_info.batch_outputs[k]])
                new_batch_outputs[k] = new_batch_outputs[k].view(
                    -1, *new_batch_outputs[k].shape[2:])
        else:
            new_batch_outputs = None
        if self.batch is not None:
            raise NotImplementedError
        return self.__class__(new_batch_length,
                              batch_stats=new_batch_stats,
                              batch_outputs=new_batch_outputs)

    def __radd__(self, i):
        if i != 0:
            raise Exception
        return self

    def __str__(self):
        step_info = ""
        first = True
        for (k,v) in sorted(self.batch_stats.items(), key=lambda kv: kv[0]):
            if not first:
                step_info += ", "
            first = False
            step_info += ("%s per instance: " % k)\
                         +str(v/self.batch_length)
        return step_info

    def to_tensor(self):
        if self.batch_outputs is not None or self.batch is not None:
            raise NotImplementedError
        list_of_floats = []
        list_of_floats.append(self.batch_length)
        for k,v in sorted(self.batch_stats.items(),
                          key=lambda kv: kv[0]):
            list_of_floats.append(v)
        return torch.tensor(list_of_floats)

    def from_tensor(self, tensor, index_iter=None):
        if index_iter is None:
            index_iter = IndexIter(0,tensor.size(0))
        self.batch_length = int(tensor[next(index_iter)].item())
        for k,v in sorted(self.batch_stats.items(),
                          key=lambda kv: kv[0]):
            self.batch_stats[k] = float(tensor[next(index_iter)].item())
        return self
