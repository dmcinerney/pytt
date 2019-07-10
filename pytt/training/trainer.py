# trains model
# train function
#   takes in (checkpoint object, loss_func, optional statistics_func, optional custom
#       step function, other options)
#   optionally saves checkpoint object
# contains a step function
#   takes in model, inputs and labels, loss_func, statistics_func
# Checkpoint object
#   contains a model, optimizer, batch_iterator, and optional val_iterator
#   contains classmethods to load each independently from a file
#   contains classmethod to load checkpoint from folder

import os
import torch
import torch.distributed as dist
#from torch.nn import DistributedDataParallel as DDP
from fairseq.legacy_distributed_data_parallel\
    import LegacyDistributedDataParallel as LDDP
from pytt.utils import MultiBatchGradMod
from pytt.logger import logger
from pytt.iteration_info import IterationInfo
from pytt.training.tracker import Tracker
from pytt.iteration_info import BatchInfo


# TODO: fix and add comments
class Trainer:
    """
    Trainer object containing model, optimizer, batch_iterator and
    val_iterator with saving and loading capabilities
    """
    def __init__(self, model, optimizer, batch_iterator, val_iterator=None,
                 val_every=1, tracker=Tracker(), checkpoint_folder=None,
                 checkpoint_every=1, batch_info_class=BatchInfo):
        self.model = model
        if dist.is_initialized() and not isinstance(self.model, LDDP):
            raise Exception
        self.optimizer = optimizer
        self.batch_iterator = batch_iterator
        self.val_iterator = val_iterator
        self.val_every = val_every
        self.tracker = tracker
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_every = checkpoint_every
        self.batch_info_class = batch_info_class

    def train(self, loss_func, statistics_func=None, grad_mod=None,
              iter_info_class=IterationInfo):
        for batch in self.batch_iterator:
            iteration_info = iter_info_class()
            self.iteration(iteration_info, batch, loss_func,
                           statistics_func=statistics_func, grad_mod=grad_mod)

    def iteration(self, iteration_info, batch, loss_func, statistics_func=None,
                  grad_mod=None):
        # record iterator info
        iteration_info.set_iterator_info(self.batch_iterator.iterator_info())
        # take val step
        if self.val_iterator is not None\
           and iteration_info.iterator_info.subbatches.last_subbatch()\
           and (iteration_info.iterator_info.batches_seen
                % self.val_every) == 0:
            self.iteration_valstep(iteration_info, loss_func,
                                   statistics_func=statistics_func)
        # take train step
        self.iteration_trainstep(iteration_info, batch, loss_func,
                                 statistics_func=statistics_func,
                                 grad_mod=grad_mod)
        # register iteration
        if self.tracker is not None:
            self.tracker.register_iteration(iteration_info)
        # save state to file
        if self.checkpoint_folder is not None\
           and iteration_info.iterator_info.take_step\
           and (iteration_info.iterator_info.batches_seen
                % self.checkpoint_every) == 0:
            self.save_state(self.checkpoint_folder)

    def iteration_trainstep(self, iteration_info, batch, loss_func,
                            statistics_func=None, grad_mod=None):
        # process training batch
        train_info = self.process_batch(batch, loss_func,
                                        statistics_func=statistics_func,
                                        enable_grad=True)
        # record training info
        iteration_info.set_train_info(self.batch_info_class(
            {k:v.item() for k,v in train_info.items()}))

        # calculate gradients
        self.calculate_grads(train_info["loss"])
        # take step if the iterator says to
        self.step(grad_mod=grad_mod, denominator=train_info["_batch_length"])

    def iteration_valstep(self, iteration_info, loss_func,
                          statistics_func=None):
        # process validation batch
        val_info = 0
        while True:
            val_info += self.batch_info_class({
                k:v.item() for k,v in\
                self.process_batch(next(self.val_iterator), loss_func,
                                   statistics_func=statistics_func,
                                   enable_grad=False).items()})
            if self.val_iterator.take_step():
                break
        # record validation info
        iteration_info.set_val_info(val_info)

    def process_batch(self, batch, loss_func, statistics_func=None,
                      enable_grad=True):
        with torch.set_grad_enabled(enable_grad):
            # run batch through the model
            outputs = self.model(**batch.get_observed())
            # calculate loss using the outputs of the model
            loss = loss_func(**outputs, **batch.get_target())
        # if statistics function is given, calculate it
        if statistics_func is not None:
            with torch.autograd.no_grad():
                stats = statistics_func(**outputs, **batch.get_target())
        step_dict = {"loss":loss, "_batch_length":torch.tensor(len(batch))}
        if statistics_func is not None:
            step_dict.update(stats)
        return step_dict

    def calculate_grads(self, loss):
        if isinstance(self.model, LDDP):
            if self.batch_iterator.take_step():
                self.model.accumulate_grads = False
            else:
                self.model.accumulate_grads = True
        loss.backward()

    def step(self, grad_mod=None, denominator=1):
        if self.batch_iterator.take_step():
            multi_batch_grad_mod = MultiBatchGradMod(denominator)
            if grad_mod is not None:
                grad_mod(list(self.model.parameters()))
            self.optimizer.step()
            self.optimizer.zero_grad()

    def save_state(self, folder):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        # save model state
        torch.save(self.model.state_dict(),
                   os.path.join(folder, 'model_state.tpkl'))
        # save optimizer state
        torch.save(self.optimizer.state_dict(),
                   os.path.join(folder, 'optimizer_state.tpkl'))
        self.batch_iterator.indices_iterator.save(
            os.path.join(folder, 'train_indices_iterator.pkl'))
        if self.val_iterator is not None:
            self.val_iterator.indices_iterator.save(
                os.path.join(folder, 'val_indices_iterator.pkl'))
        # TODO: fix this so that history is appended rather than resaved
        self.tracker.save(os.path.join(folder, 'tracker.pkl'))
