import os
import torch
import torch.distributed as dist
try:
    from fairseq.legacy_distributed_data_parallel\
        import LegacyDistributedDataParallel as LDDP
except ImportError:
    LDDP = type(None)
from pytt.distributed import log_bool
from pytt.logger import logger
from pytt.iteration_info import IterationInfo
from pytt.training.tracker import Tracker
from pytt.iteration_info import BatchInfo
from pytt.utils import MultiBatchGradMod, indent, write_pickle, get_random_state
from pytt.progress_bar import ProgressBar

class Trainer:
    """
    Trainer object containing model, optimizer, train_iterator and optionally,
    a val_iterator, and a tracker.  It also optionally saves everything along
    with a ranodm state to a checkpoint_folder.  It can use custom batch info
    classes and progress_bar classes, and the frequency of validating,
    checkpointing, and printing can be controlled.
    """
    def __init__(self, model, optimizer, train_iterator, val_iterator=None,
                 tracker=None, checkpoint_folder=None,
                 batch_info_class=BatchInfo, val_every=1,
                 checkpoint_every=1, print_every=1, pbar=None,
                 keep_subbatch_outputs=True):
        if dist.is_initialized() and LDDP is None:
            raise Exception
        self.model = model
        if dist.is_initialized() and not isinstance(self.model, LDDP):
            raise Exception
        self.optimizer = optimizer
        self.train_iterator = train_iterator
        self.pbar = pbar if pbar is not None else ProgressBar()
        self.val_iterator = val_iterator
        self.tracker = Tracker(
            checkpoint_folder=os.path.join(
                checkpoint_folder, 'tensorboard'),
            purge_step=self.train_iterator.iterator_info().batches_seen)\
            if tracker is None else tracker
        self.checkpoint_folder = checkpoint_folder
        self.batch_info_class = batch_info_class
        self.val_every = val_every
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.keep_subbatch_outputs = keep_subbatch_outputs

    def train(self, loss_func, statistics_func=None, grad_mod=None,
              iter_info_class=IterationInfo, use_pbar=True):
        """
        Trains the model by calling iteration until iteration throws a
        StopIteration Exception.  It uses the loss_func (not optional) to
        perform optimization and the statistics_func to return relevant batch
        statistics to keep track of throughout training.  This only returns a
        dictionary of floats, and cannot be used to record any other type of
        data.  Information from the iterator_info, and the loss and statistics
        numbers from each batch (optionally also the val batch) are collected
        on the iteration_info object which is initialized every iteration using
        the iter_info_class. Optionally, one can pass in a customized
        iter_info_class. The grad_mod function can be used to modify the
        gradient before each gradient step is taken. The progress bar is used
        unless use_pbar is specified False.
        """
        if use_pbar:
            self.pbar.enter(total=len(self.train_iterator.indices_iterator),
                initial=self.train_iterator.iterator_info().batches_seen)
        try:
            while True:
                iteration_info = iter_info_class()
                self.iteration(iteration_info, loss_func,
                    statistics_func=statistics_func, grad_mod=grad_mod)
        except StopIteration:
            if use_pbar:
                self.pbar.exit()
            self.tracker.close()

    def iteration(self, iteration_info, loss_func, statistics_func=None,
                  grad_mod=None):
        batches_seen = self.train_iterator.iterator_info().batches_seen
        # take val step
        if self.val_iterator is not None\
           and ((batches_seen+1) % self.val_every) == 0:
            self.iteration_valstep(iteration_info, loss_func,
                                   statistics_func=statistics_func)
        # take train step
        self.iteration_trainstep(iteration_info, loss_func,
                                 statistics_func=statistics_func,
                                 grad_mod=grad_mod)
        # record iterator info
        iteration_info.set_iterator_info(self.train_iterator.iterator_info())
        # register iteration info with the tracker
        self.tracker.register_iteration(iteration_info)
        # print tracker info
        if self.recurring_bool(iteration_info, self.print_every)\
           and log_bool():
            logger.log(str(self.tracker))
        # save state to file
        if self.checkpoint_folder is not None\
           and self.recurring_bool(iteration_info, self.checkpoint_every)\
           and log_bool():
            logger.log("saving checkpoint to %s, batches_seen: %i" %
                       (self.checkpoint_folder,
                        iteration_info.iterator_info.batches_seen))
            self.save_state(self.checkpoint_folder)
        # update progress bar
        self.pbar.update()

    def recurring_bool(self, iteration_info, every):
        return (iteration_info.iterator_info.batches_seen
                % every) == 0\
               or (iteration_info.iterator_info.batches_seen
                   == iteration_info.iterator_info.total_batches)

    def iteration_trainstep(self, iteration_info, loss_func,
                            statistics_func=None, grad_mod=None):
        train_info = 0
        # iterate through all the subbatches in a batch, accumulating gradients
        while True:
            # process training subbatch
            train_stats, train_outputs = self.process_batch(next(self.train_iterator),
                loss_func, statistics_func=statistics_func, enable_grad=True)
            # get iterator_info from iterator
            iterator_info = self.train_iterator.iterator_info()
            # calculate and accumulate gradients
            self.calculate_grads(train_stats["loss"])
            # accumulate batch info
            train_info += self.batch_info_class(
                {k:v.item() for k,v in train_stats.items()},
                batch_outputs={k:v.detach()
                               for k,v in train_outputs.items()},
                max_num_instance_outputs=
                    iterator_info.subbatches.samples_in_fullbatch
                    if train_outputs is not None else None)
            # log subbatch info
            if ((iterator_info.batches_seen
                 + int(not self.train_iterator.take_step()))
                % self.print_every) == 0:
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
                  denominator=train_info.batch_stats["_batch_length"])


    def iteration_valstep(self, iteration_info, loss_func,
                          statistics_func=None):
        val_info = 0
        # iterate through all the subbatches in a batch
        while True:
            # process subbatch and accumulate validation info
            val_stats, val_outputs = self.process_batch(
                next(self.val_iterator), loss_func,
                statistics_func=statistics_func, enable_grad=False)
            iterator_info = self.val_iterator.iterator_info()
            val_info += self.batch_info_class({
                k:v.item() for k,v in val_stats.items()},
                batch_outputs={k:v.detach()
                               for k,v in val_outputs.items()},
                max_num_instance_outputs=
                    iterator_info.subbatches.samples_in_fullbatch
                    if val_outputs is not None else None)

            # end loop if the iterator says to take a step
            if self.val_iterator.take_step():
                break
        # record validation info
        iteration_info.set_val_info(val_info)

    def process_batch(self, batch, loss_func, statistics_func=None,
                      enable_grad=True):
        if self.tracker.needs_graph:
            self.tracker.add_graph(self.model, batch)
        # enable or disable gradients
        with torch.set_grad_enabled(enable_grad):
            # run batch through the model
            outputs = self.model(**batch.get_observed())
            # calculate loss using the outputs of the model and the batch
            # targets
            loss = loss_func(**outputs, **batch.get_target())
        # if statistics function is given, calculate it without gradients
        if statistics_func is not None:
            with torch.autograd.no_grad():
                stats = statistics_func(**outputs, **batch.get_target())
        # create the dictionary to return
        return_dict = {"loss":loss, "_batch_length":torch.tensor(len(batch))}
        # includes statistics in the return_dict
        if statistics_func is not None:
            return_dict.update(stats)
        if not self.keep_subbatch_outputs:
            outputs = None
        return return_dict, outputs

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
        # save tracker
        # TODO: fix this so that history is appended rather than resaved
        self.tracker.save(os.path.join(folder, 'tracker.pkl'))
