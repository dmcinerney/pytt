import os
import torch
import torch.distributed as dist
try:
    from fairseq.legacy_distributed_data_parallel\
        import LegacyDistributedDataParallel as LDDP
except ImportError:
    LDDP = type(None)
from pytt.logger import logger
from pytt.iteration_info import IterationInfo
from pytt.training.tracker import Tracker
from pytt.iteration_info import BatchInfo
from pytt.utils import MultiBatchGradMod, indent, write_pickle, get_random_state

class Trainer:
    """
    Trainer object containing model, optimizer, train_iterator and optionally,
    a val_iterator, and a tracker.  It also optionally saves everything along
    with a random state to a checkpoint_folder.  It can use custom batch info
    classes and progress_bar classes, and the frequency of validating,
    checkpointing, and printing can be controlled with the tracker object.
    """
    def __init__(self, model, optimizer, train_iterator, val_iterator=None,
                 tracker=None, batch_info_class=BatchInfo, val_every=1):
        if dist.is_initialized() and LDDP is None:
            raise Exception
        self.model = model
        if dist.is_initialized() and not isinstance(self.model, LDDP):
            raise Exception
        self.optimizer = optimizer
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.tracker = Tracker(
            purge_step=self.train_iterator.iterator_info().batches_seen)\
            if tracker is None else tracker
        self.batch_info_class = batch_info_class
        self.val_every = val_every

    def train(self, loss_func, grad_mod=None, iter_info_class=IterationInfo):
        """
        Trains the model by calling iteration until iteration throws a
        StopIteration Exception.
        TODO: Need to fill this out further.
        """
        self.tracker.enter(total=len(self.train_iterator.indices_iterator),
            initial=self.train_iterator.iterator_info().batches_seen)
        try:
            while True:
                iteration_info = iter_info_class()
                self.iteration(iteration_info, loss_func, grad_mod=grad_mod)
        except StopIteration:
            self.tracker.close()

    def iteration(self, iteration_info, loss_func, grad_mod=None):
        batches_seen = self.train_iterator.iterator_info().batches_seen
        # take val step
        if self.val_iterator is not None\
           and ((batches_seen+1) % self.val_every) == 0:
            self.iteration_valstep(iteration_info, loss_func)
        # take train step
        self.iteration_trainstep(iteration_info, loss_func, grad_mod=grad_mod)
        # record iterator info
        iteration_info.set_iterator_info(self.train_iterator.iterator_info())
        # register iteration info with the tracker
        self.tracker.register_iteration(iteration_info, self)

    def iteration_trainstep(self, iteration_info, loss_func, grad_mod=None):
        train_info = 0
        # iterate through all the subbatches in a batch, accumulating gradients
        while True:
            # process training subbatch
            loss, batch_info = self.process_batch(
                next(self.train_iterator), loss_func, enable_grad=True)
            # get iterator_info from iterator
            iterator_info = self.train_iterator.iterator_info()
            # calculate and accumulate gradients
            self.calculate_grads(loss)
            # accumulate batch info
            train_info += batch_info
            # log subbatch info
            if ((iterator_info.batches_seen
                 + int(not self.train_iterator.take_step()))
                % self.tracker.print_every) == 0:
                logger.log(indent(iterator_info.subbatch_str(),
                                  "        "), verbosity=2)
            # end loop if the iterator says to take a gradient step
            if self.train_iterator.take_step():
                break
        # record training info
        iteration_info.set_train_info(train_info)
        # take a gradient step, with loss summed over all subbatches on all
        # devices, dividing by the number of instances
        self.step(grad_mod=grad_mod,
                  denominator=train_info.batch_length)


    def iteration_valstep(self, iteration_info, loss_func):
        val_info = 0
        # iterate through all the subbatches in a batch
        while True:
            # process subbatch and accumulate validation info
            _, batch_info = self.process_batch(
                next(self.val_iterator), loss_func, enable_grad=False)
            # get iterator_info from iterator
            iterator_info = self.val_iterator.iterator_info()
            # accumulate batch info
            val_info += batch_info

            # end loop if the iterator says to take a step
            if self.val_iterator.take_step():
                break
        # record validation info
        iteration_info.set_val_info(val_info)

    def process_batch(self, batch, loss_func, enable_grad=True):
        if self.tracker.needs_graph:
            self.tracker.add_graph(self.model, batch)
        # enable or disable gradients
        with torch.set_grad_enabled(enable_grad):
            # run batch through the model
            outputs = self.model(**batch.get_observed())
            # calculate loss using the outputs of the model and the batch
            # targets
            kwargs = {**outputs, **batch.get_target()}
            loss = loss_func(**kwargs)
        # create batch_info object
        return loss, self.batch_info_class(len(batch), batch=batch, batch_outputs=kwargs, loss=loss)

    def calculate_grads(self, loss):
        # if the model is distributed (a fairseq Legacy Distributed Data
        # Parallel module), if the iterator says to take a step, set
        # accumulated_grads to False, allowing synching of the gradients during
        # the backward pass.  Otherwise, set it to True to allow local
        # accumulation of gradients without synching messing up the calculation
        if isinstance(self.model, LDDP):
            if self.train_iterator.take_step():
                self.model.accumulate_grads = False
            else:
                self.model.accumulate_grads = True
        loss.backward()

    def step(self, grad_mod=None, denominator=1):
        # create a multi-batch grad mod object, which multiplies the gradients
        # by the number of devices, and devides it by the denominator
        multi_batch_grad_mod = MultiBatchGradMod(denominator)
        if grad_mod is not None:
            multi_batch_grad_mod(self.model.parameters())
            grad_mod(self.model.parameters())
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_state(self, folder):
        # save random state
        write_pickle(get_random_state(),
                     os.path.join(folder, 'random_state.pkl'))
        # save model state
        torch.save(self.model.state_dict()\
                   if not isinstance(self.model, LDDP) else\
                   self.model.module.state_dict(),
                   os.path.join(folder, 'model_state.tpkl'))
        # save optimizer state
        torch.save(self.optimizer.state_dict(),
                   os.path.join(folder, 'optimizer_state.tpkl'))
        # save train iterator
        self.train_iterator.indices_iterator.save(
            os.path.join(folder, 'train_indices_iterator.pkl'))
        # save val iterator
        if self.val_iterator is not None:
            self.val_iterator.indices_iterator.save(
                os.path.join(folder, 'val_indices_iterator.pkl'))
