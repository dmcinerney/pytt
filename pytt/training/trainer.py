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
from tqdm import tqdm
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
from pytt.utils import indent


# TODO: fix and add comments
class Trainer:
    """
    Trainer object containing model, optimizer, batch_iterator and
    val_iterator with saving and loading capabilities
    """
    def __init__(self, model, optimizer, batch_iterator, val_iterator=None,
                 tracker=Tracker(), checkpoint_folder=None,
                 batch_info_class=BatchInfo, use_pbar=True, val_every=1,
                 checkpoint_every=1, print_every=1):
        self.model = model
        if dist.is_initialized() and not isinstance(self.model, LDDP):
            raise Exception
        self.optimizer = optimizer
        self.batch_iterator = batch_iterator
        if (not dist.is_initialized()
            or (dist.is_initialized() and dist.get_rank() == 0))\
           and use_pbar:
            self.pbar = tqdm(total=len(self.batch_iterator.indices_iterator), mininterval=1)
        else:
            self.pbar = None
        self.val_iterator = val_iterator
        self.tracker = tracker
        self.checkpoint_folder = checkpoint_folder
        self.batch_info_class = batch_info_class
        self.val_every = val_every
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

    def train(self, loss_func, statistics_func=None, grad_mod=None,
              iter_info_class=IterationInfo):
        logger.set_progress_bar(self.pbar)
        try:
            while True:
                iteration_info = iter_info_class()
                self.iteration(iteration_info, loss_func,
                    statistics_func=statistics_func, grad_mod=grad_mod)
        except StopIteration:
            pass

    def iteration(self, iteration_info, loss_func, statistics_func=None,
                  grad_mod=None):
        # take val step
        batches_seen = self.batch_iterator.iterator_info().batches_seen
        if self.val_iterator is not None\
           and ((batches_seen+1) % self.val_every) == 0:
            self.iteration_valstep(iteration_info, loss_func,
                                   statistics_func=statistics_func)
        # take train step
        self.iteration_trainstep(iteration_info, loss_func,
                                 statistics_func=statistics_func,
                                 grad_mod=grad_mod)
        # record iterator info
        iteration_info.set_iterator_info(self.batch_iterator.iterator_info())
        # register iteration
        self.tracker.register_iteration(iteration_info)
        if (iteration_info.iterator_info.batches_seen
            % self.print_every) == 0:
            self.tracker.log_last_iteration()
        # save state to file
        if self.checkpoint_folder is not None\
           and iteration_info.iterator_info.take_step\
           and (iteration_info.iterator_info.batches_seen
                % self.checkpoint_every) == 0:
            self.save_state(self.checkpoint_folder)

    def iteration_trainstep(self, iteration_info, loss_func,
                            statistics_func=None, grad_mod=None):
        train_info = 0
        while True:
            # process training batch
            train_info_dict = self.process_batch(self.next_training_batch(), loss_func,
                                                 statistics_func=statistics_func,
                                                 enable_grad=True)
            iterator_info = self.batch_iterator.iterator_info()
            if ((iterator_info.batches_seen + int(not self.batch_iterator.take_step()))
                % self.print_every) == 0:
                logger.log(indent(iterator_info.subbatch_str(),
                                  "        "), verbosity=2)
            # calculate gradients
            self.calculate_grads(train_info_dict["loss"])
            train_info += self.batch_info_class(
                {k:v.item() for k,v in train_info_dict.items()})
            if self.batch_iterator.take_step():
                break
        # record training info
        iteration_info.set_train_info(train_info)
        # take step
        self.step(grad_mod=grad_mod, denominator=train_info.batch_info_dict["_batch_length"])

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

    def next_training_batch(self):
        batch = next(self.batch_iterator)
        if self.pbar is not None\
           and self.batch_iterator.take_step():
            self.pbar.update()
        return batch
