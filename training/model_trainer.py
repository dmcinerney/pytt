# trains model
# train function
#   takes in (checkpoint object, loss_func, optional error_func, optional custom
#       step function, other options)
#   optionally saves checkpoint object
# contains a step function
#   takes in model, inputs and labels, loss_func, error_func
# Checkpoint object
#   contains a model, optimizer, batch_iterator, and optional val_iterator
#   contains classmethods to load each independently from a file
#   contains classmethod to load checkpoint from folder

import torch
from utils import MultiBatchGradMod
from logger import Logger
from fairseq.legacy_distributed_data_parallel\
    import LegacyDistributedDataParallel as LDDP
from distributed.distributed import collect_tensors_on_rank0
import copy


# TODO: fix and add comments
class Trainer:
    """
    Trainer object containing model, optimizer, batch_iterator and
    val_iterator with saving and loading capabilities
    """
    @classmethod
    def load(cls, folder):
        raise NotImplementedError

    def __init__(self, model, optimizer, batch_iterator, val_iterator=None,
                 val_every=1):
        self.model = model
        self.optimizer = optimizer
        self.batch_iterator = batch_iterator
        self.val_iterator = val_iterator
        self.val_every = val_every
        self.history = []
        self.accumulated_loss = 0
        self.accumulated_error = 0

    def train(self, loss_func, error_func=None, grad_mod=None):
        for batch in self.batch_iterator:
            self.iteration(batch, loss_func, error_func=error_func,
                           grad_mod=grad_mod)

    def iteration(self, batch, loss_func, error_func=None, grad_mod=None):
        iteration_info = {}
        iterator_info = self.batch_iterator.iterator_info()
        iteration_info["iterator_info"] = iterator_info
        train_process_dict = self.process_batch(batch, loss_func,
                                                error_func=error_func,
                                                enable_grad=True)
        iteration_info["train_process_dict"] = train_process_dict
        self.step(train_process_dict["loss"], grad_mod=grad_mod)
        if self.val_iterator is not None\
           and (iterator_info["batches_seen"] % val_every) == 0:
            val_batch = next(self.val_iterator)
            val_process_dict = self.process_batch(val_batch, loss_func,
                                                  error_func=error_func,
                                                  enable_grad=False)
            iteration_info["val_process_dict"] = val_process_dict
        self.register_iteration(iteration_info)

    def process_batch(self, batch, loss_func, error_func=None,
                      enable_grad=True):
        with torch.set_grad_enabled(enable_grad):
            outputs = self.model(batch)
            loss = loss_func(**outputs)
        if error_func is not None:
            with torch.autograd.no_grad():
                error = error_func(**outputs)
        step_dict = {"loss":loss}
        if error_func is not None:
            step_dict["error"] = error
        return step_dict

    def step(self, loss, grad_mod=None):
        if isinstance(self.model, LDDP):
            if self.batch_iterator.take_step():
                self.model.accumulate_grads = False
            else:
                self.model.accumulate_grads = True
        loss.backward()
        if self.batch_iterator.take_step():
            multi_batch_grad_mod = MultiBatchGradMod(
                self.batch_iterator.iterator_info()["samples_in_batch"])
            if grad_mod is not None:
                grad_mod(list(self.model.parameters()))
            self.optimizer.step()
            self.optimizer.zero_grad()

    def register_iteration(self, iteration_info):
        iteration_log = {}
        
        # put everything into the log, calling item on scalar tensors
        iteration_log["iterator_info"] = iteration_info["iterator_info"]
        iteration_log["train_process_dict"] = {
            k:v.item() for k,v in iteration_info["train_process_dict"].items()}
        if "val_process_dict" in iteration_info.keys():
            iteration_log["val_process_dict"] = {
                k:v.item() for k,v in iteration_info["val_process_dict"].items()}
        
        # keep track of accumulated loss and error per batch
        self.accumulated_loss += iteration_log["train_process_dict"]["loss"]
        iteration_log["train_process_dict"]["accumulated_loss"] =\
            self.accumulated_loss
        if "error" in iteration_log["train_process_dict"].keys():
            self.accumulated_error += iteration_log["train_process_dict"]["error"]
            iteration_log["train_process_dict"]["accumulated_error"] =\
                self.accumulated_error
        if self.batch_iterator.take_step():
            self.accumulated_loss = 0
            self.accumulated_error = 0
        
        # register log in the history and print out log, keeping track of distributed logs
        if torch.distributed.is_initialized():
            collected = self.collect_logs_on_rank0(iteration_log)
            if collected is not None:
                coalesced_log, all_logs = collected
                self.history.append((coalesced_log, all_logs))
                self.log_iteration(coalesced_log)
            else:
                self.history.append(iteration_log)
        else:
            self.history.append(iteration_log)
            self.log_iteration(iteration_log)

    def log_iteration(self, iteration_log):
        if "subbatches_seen" in iteration_log["iterator_info"].keys():
            subbatch_info = "\tbatches_seen: "\
                +str(iteration_log["iterator_info"]["batches_seen"])\
                +", samples_seen: "\
                +str(iteration_log["iterator_info"]["samples_seen"])\
                +", subbatches_seen: "\
                +str(iteration_log["iterator_info"]["subbatches_seen"])\
                +", samples_in_batch_seen: "\
                +str(iteration_log["iterator_info"]["samples_in_batch_seen"])
            Logger.log(subbatch_info)
        if self.batch_iterator.take_step():
            step_info = "batches_seen: "\
                +str(iteration_log["iterator_info"]["batches_seen"])\
                +", samples_seen: "\
                +str(iteration_log["iterator_info"]["samples_seen"])\
                +", accumulated_loss: "\
                +str(iteration_log["train_process_dict"]["accumulated_loss"])
            Logger.log(step_info)
    
    def collect_logs_on_rank0(self, iteration_log):
        tensors = collect_tensors_on_rank0(self.iterlog_to_tensor(iteration_log))
        if tensors is None:
            return None
        logs = [self.tensor_to_iterlog(tensor, iteration_log) for tensor in tensors]
        coalesced_log = copy.deepcopy(logs[0])
        add_along_keys = []
        for k1 in iteration_log.keys():
            if k1 == "iterator_info":
                if "samples_in_batch_seen" in iteration_log[k1].keys():
                    add_along_keys.append((k1, "samples_in_batch_seen"))
                continue
            for k2 in iteration_log[k1].keys():
                add_along_keys.append((k1, k2))
        for k1,k2 in add_along_keys:
            coalesced_log[k1][k2] = 0
            for log in logs:
                coalesced_log[k1][k2] += log[k1][k2]
        return coalesced_log, logs

    def iterlog_to_tensor(self, iteration_log):
        list_of_floats = []
        keys = ['iterator_info', 'train_process_dict']
        if self.val_iterator is not None:
            keys.apppend('val_process_dict')
        for iter_log_key in keys:
            list_of_floats += [
                v for k,v in sorted(iteration_log[iter_log_key].items(),
                                    key=lambda kv: kv[0])
            ]
        return torch.tensor(list_of_floats)

    def tensor_to_iterlog(self, tensor, iteration_log_template):
        iteration_log = copy.deepcopy(iteration_log_template)
        tensor_iter = iter(tensor)
        keys = ['iterator_info', 'train_process_dict']
        if self.val_iterator is not None:
            keys.apppend('val_process_dict')
        for iter_log_key in keys:
            cast = int if iter_log_key is 'iterator_info' else float
            for k,v in sorted(iteration_log[iter_log_key].items(),
                              key=lambda kv: kv[0]):
                iteration_log[iter_log_key][k] = cast(next(tensor_iter).item())
        return iteration_log

    def save(self, folder):
        raise NotImplementedError
