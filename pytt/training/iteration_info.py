import torch
import torch.distributed as dist
from pytt.logger import logger


# TODO: add comments
class IterationInfo:
    def __init__(self, batch_info_class=None):
        self.iterator_info = None
        self.train_info = None
        self.val_info = None
        self.subbatch_info_list = None
        self.batch_info_class = BatchInfo\
                                if batch_info_class is None else\
                                batch_info_class

    def set_iterator_info(self, iterator_info):
        self.iterator_info = iterator_info

    def set_train_info(self, batch_info):
        self.train_info = self.batch_info_class(batch_info)

    def check_initialized(self):
        return self.iterator_info is not None and self.train_info is not None

    def set_val_info(self, batch_info):
        self.val_info = self.batch_info_class(step_info)

    def set_subbatch_info_list(self, subbatch_info_list):
        self.subbatch_info_list = subbatch_info_list

    def add_process_batch_info(self, process_batch_info1, process_batch_info2):
        process_batch_info = {}
        for k in set(process_batch_info1.keys()).union(
                     process_batch_info2.keys()):
            process_batch_info[k] = process_batch_info1[k]\
                                    + process_batch_info2[k]
        return process_batch_info

    def __add__(self, iteration_info):
        if not (self.check_initialized()
                and iteration_info.check_initialized()):
            raise Exception
        new_iteration_info = self.__class__(
            batch_info_class=self.batch_info_class)

        # set iteration_info
        new_iteration_info.set_iterator_info(
            self.iterator_info+iteration_info.iterator_info)

        # set train_info
        new_iteration_info.train_info = self.train_info\
                                        +iteration_info.train_info

        # set val_info
        if self.val_info is not None and iteration_info.val_info is not None:
            new_iteration_info.val_info = self.val_info\
                                          +iteration_info.val_info

        # set subbatch_info_list
        subbatch_info_list = []
        for iter_info in (self, iteration_info):
            if iter_info.subbatch_info_list is None:
                subbatch_info_list.append(iter_info)
            else:
                subbatch_info_list.extend(iter_info.subbatch_info_list)
        new_iteration_info.set_subbatch_info_list(subbatch_info_list)

        return new_iteration_info

    def __radd__(self, i):
        if i != 0:
            raise Exception
        return self

    def log_iteration(self, full_batch=True):
        if not full_batch:
            self.log_subbatch()
        else:
            self.log_fullbatch()

    def log_subbatch(self):
        logger.log(self.iterator_info.subbatch_str(), verbosity=2)

    def log_fullbatch(self):
        step_info = str(self.iterator_info)
        step_info += "\n  TRAIN\n"+str(self.train_info)
        if self.val_info is not None:
            step_info += "\n  VAL\n"+str(self.val_info)
        logger.log(step_info)

    def to_tensor(self):
        tensors = []
        tensors.append(self.iterator_info.to_tensor().float())
        tensors.append(self.train_info.to_tensor())
        if self.val_info is not None:
            tensors.append(self.val_info.to_tensor())
        if self.subbatch_info_list is not None:
            for subbatch_iteration_info in self.subbatch_info_list:
                tensors.append(subbatch_iteration_info.to_tensor())
        return torch.cat(tensors, 0)

    def from_tensor(self, tensor, isiter=False):
        if not isiter:
            tensor_iter = iter(tensor)
        else:
            tensor_iter = tensor
        self.iterator_info.from_tensor(tensor_iter, isiter=True)
        self.train_info.from_tensor(tensor_iter, isiter=True)
        if self.val_info is not None:
            self.train_info.from_tensor(tensor_iter, isiter=True)
        if self.subbatch_info_list is not None:
            for subbatch_iteration_info in self.subbatch_info_list:
                subbatch_iteration_info.from_tensor(tensor_iter, isiter=True)
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

    def __str__(self):
        step_info = "    "
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