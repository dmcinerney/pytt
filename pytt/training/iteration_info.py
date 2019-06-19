import copy
import torch
import torch.distributed as dist
from pytt.logger import logger


# TODO: add comments
class IterationInfo:
    def __init__(self):
        self.iterator_info = None
        self.train_info = None
        self.val_info = None
        self.subbatch_info_list = None

    def set_iterator_info(self, iterator_info):
        self.iterator_info = iterator_info

    def set_train_info(self, train_info):
        self.train_info = train_info

    def check_initialized(self):
        return self.iterator_info is not None and self.train_info is not None

    def set_val_info(self, val_info):
        self.val_info = val_info

    def set_subbatch_info_list(self, subbatch_info_list):
        self.subbatch_info_list = subbatch_info_list

    def __add__(self, iteration_info):
        if not (self.check_initialized()
                and iteration_info.check_initialized()):
            raise Exception
        new_iteration_info = IterationInfo()

        # set iteration_info
        iterator_info = {}
        iterator_info["batches_seen"] = \
            iteration_info.iterator_info["batches_seen"]
        iterator_info["samples_seen"] = \
            iteration_info.iterator_info["samples_seen"]
        iterator_info["take_step"] = self.iterator_info["take_step"]\
            or iteration_info.iterator_info["take_step"]
        if "samples_in_subbatch" in set(self.iterator_info.keys()).union(
                                    iteration_info.iterator_info.keys()):
            iterator_info["samples_in_subbatch"] = \
                self.iterator_info["samples_in_subbatch"]\
                + iteration_info.iterator_info["samples_in_subbatch"]
        if "rank" in set(self.iterator_info.keys()).union(
                     iteration_info.iterator_info.keys()):
            if self.iterator_info["rank"] == self.iterator_info["rank"]:
                iterator_info["rank"] = self.iterator_info["rank"]
            iterator_info["worldsize"] = self.iterator_info["worldsize"]
            iterator_info["samples_per_process"] = \
                self.iterator_info["samples_per_process"]
        iterator_info["samples_in_batch"] = \
            self.iterator_info["samples_in_batch"]
        new_iteration_info.set_iterator_info(iterator_info)

        # set train_info
        train_info = {}
        for k in set(self.train_info.keys()).union(
                 iteration_info.train_info.keys()):
            train_info[k] = self.train_info[k] + iteration_info.train_info[k]
        new_iteration_info.set_train_info(train_info)

        # set val_info
        if self.val_info is not None and iteration_info.val_info is not None:
            val_info = {}
            for k in set(self.val_info.keys()).union(
                     iteration_info.val_info.keys()):
                val_info[k] = self.val_info[k] + iteration_info.val_info[k]
            new_iteration_info.set_val_info(val_info)

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
        base = "batches_seen: "\
            +str(self.iterator_info["batches_seen"])\
            +", samples_seen: "\
            +str(self.iterator_info["samples_seen"])
        if not full_batch:
            is_multibatchperprocess = \
                "samples_in_subbatch" in self.iterator_info.keys()
            is_multiprocess = dist.is_initialized()
            if not is_multibatchperprocess and not is_multiprocess:
                return
            subbatch_info = "\t"+base
            if is_multibatchperprocess:
                subbatch_info += ", subbatches_seen: "\
                    +str(len(self.subbatch_info_list)
                         if self.subbatch_info_list is not None else 1)
            subbatch_info += ", samples_in_batch_seen: "\
                + (str(self.iterator_info["samples_in_subbatch"])
                   if is_multibatchperprocess else
                   str(self.iterator_info["samples_per_process"]))
            if is_multiprocess:
                subbatch_info += ", rank: "\
                    +str(self.iterator_info["rank"])
            logger.log(subbatch_info, verbosity=2)
        else:
            num_instances = self.iterator_info["samples_in_batch"]
            step_info = base\
                +", train batch size: "+str(num_instances)\
                +", train loss per instance: "\
                +str(self.train_info["loss"]/num_instances)
            if "error" in self.train_info.keys():
                step_info += ", train error per instance: "\
                    +str(self.train_info["error"]/num_instances)
            if self.val_info is not None:
                step_info += ", (implement printing val info)"
            logger.log(step_info)

    def to_tensor(self):
        list_of_floats = []
        list_of_floats += [
            v for k,v in sorted(self.iterator_info.items(),
                                key=lambda kv: kv[0])
        ]
        list_of_floats += [
            v for k,v in sorted(self.train_info.items(),
                                key=lambda kv: kv[0])
        ]
        if self.val_info is not None:
            list_of_floats += [
                v for k,v in sorted(self.val_info.items(),
                                    key=lambda kv: kv[0])
            ]
        if self.subbatch_info_list is not None:
            for subbatch_iteration_info in self.subbatch_info_list:
                list_of_floats += [
                    i for i in subbatch_iteration_info.to_tensor()]
        return torch.tensor(list_of_floats)

    def from_tensor(self, tensor, isiter=False):
        iteration_info = copy.deepcopy(self)
        if not isiter:
            tensor_iter = iter(tensor)
        else:
            tensor_iter = tensor
        for k,v in sorted(iteration_info.iterator_info.items(),
                          key=lambda kv: kv[0]):
            iteration_info.iterator_info[k] = int(next(tensor_iter).item())
        for k,v in sorted(iteration_info.train_info.items(),
                          key=lambda kv: kv[0]):
            iteration_info.train_info[k] = float(next(tensor_iter).item())
        if iteration_info.val_info is not None:
            for k,v in sorted(iteration_info.val_info.items(),
                              key=lambda kv: kv[0]):
                iteration_info.val_info[k] = float(next(tensor_iter).item())
        if iteration_info.subbatch_info_list is not None:
            new_subbatch_info_list = []
            for subbatch_iteration_info in iteration_info.subbatch_info_list:
                subbatch_iteration_info = subbatch_iteration_info.from_tensor(
                    tensor_iter, isiter=True)
            iteration_info.set_subbatch_info_list(new_subbatch_info_list)
        return iteration_info