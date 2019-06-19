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
from fairseq.legacy_distributed_data_parallel\
    import LegacyDistributedDataParallel as LDDP
from pytt.utils import MultiBatchGradMod
from pytt.logger import logger
from pytt.training.iteration_info import IterationInfo
from pytt.training.history import History


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
                 val_every=1, history=History()):
        self.model = model
        self.optimizer = optimizer
        self.batch_iterator = batch_iterator
        self.val_iterator = val_iterator
        self.val_every = val_every
        self.history = history

    def train(self, loss_func, error_func=None, grad_mod=None,
              iter_info_class=IterationInfo):
        for batch in self.batch_iterator:
            iteration_info = iter_info_class()
            self.iteration(iteration_info, batch, loss_func,
                           error_func=error_func, grad_mod=grad_mod)
            if self.history is not None:
                self.history.register_iteration(iteration_info)

    def iteration(self, iteration_info, batch, loss_func, error_func=None,
                  grad_mod=None):
        # record iterator info
        iteration_info.set_iterator_info(self.batch_iterator.iterator_info())

        # process training batch
        train_info = self.process_batch(batch, loss_func, error_func=error_func,
                                        enable_grad=True)
        # record training info
        iteration_info.set_train_info(
            {k:v.item() for k,v in train_info.items()})

        # update with gradients if the iterator says to
        self.step(train_info["loss"], grad_mod=grad_mod)

        # record validation info if val_iterator is given and it is the right
        #   step
        if self.val_iterator is not None\
           and (iterator_info["batches_seen"] % val_every) == 0:
            # process validation batch
            val_info = self.process_batch(next(self.val_iterator), loss_func,
                                          error_func=error_func,
                                          enable_grad=False)
            # record validation info
            iteration_info.set_val_info(
                {k:v.item() for k,v in val_info.items()})

    def process_batch(self, batch, loss_func, error_func=None,
                      enable_grad=True):
        with torch.set_grad_enabled(enable_grad):
            # run batch through the model
            outputs = self.model(batch)
            # calculate loss using the outputs of the model
            loss = loss_func(**outputs)
        # if error function is given, calculate error
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

    def save(self, folder):
        raise NotImplementedError
