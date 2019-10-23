import os
import socket
from datetime import datetime
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from pytt.utils import read_pickle, write_pickle
from pytt.distributed import collect_obj_on_rank0, log_bool
from pytt.progress_bar import ProgressBar
from pytt.logger import logger

class Tracker:
    """
    Tracker object that creates a list (history) where each element is the info
    from one training iteration, handling info objects distributed across
    multiple devices.  Contains saving and loading functionality for use during
    checkpoint.  Also contains a string function which can be used for logging
    an iteration during training.
    """
    def __init__(self, pbar=None, print_every=1, checkpoint_every=1, checkpoint_folder=None, tensorboard_every=1, summary_writers=['train', 'val'], needs_graph=True, purge_step=None):
        self.print_every = print_every
        self.iteration_info = None
        if not log_bool():
            self.needs_graph = needs_graph
            return
        self.pbar = pbar if pbar is not None else ProgressBar()
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = checkpoint_folder
        self.tensorboard_every = tensorboard_every
        # set up tensorboard
        if checkpoint_folder is None:
            datetime_machine = datetime.now().strftime('%b%d_%H-%M-%S')\
                               + '_' + socket.gethostname()
            tensorboard_folder = os.path.join('runs', datetime_machine)
        else:
            tensorboard_folder = os.path.join(checkpoint_folder, 'tensorboard')
        self.summary_writers = {k:
            SummaryWriter(log_dir=tensorboard_folder+'/'+k,
                          purge_step=purge_step)
            for k in summary_writers}
        self.needs_graph = needs_graph

    def add_graph(self, model, batch):
        if not log_bool():
            return
        keys,values = list(zip(*((k,v)
            for k,v in batch.get_observed().items())))
        model = ModelWrapper(model, keys)
        for writer in self.summary_writers.values():
            writer.add_graph(model, values)
        self.needs_graph = False

    def register_iteration(self, iteration_info, trainer):
        self.iteration_info = iteration_info
        if dist.is_initialized():
            collected = collect_obj_on_rank0(
                self.iteration_info,
                ranks=self.iteration_info.iterator_info.subbatches.get_ranks())
            if collected is not None:
                self.iteration_info = sum(collected)
            else:
                self.iteration_info = None
        if log_bool():
            if self.recurring_bool(iteration_info, self.print_every):
                logger.log(str(self.iteration_info))
            if len(self.summary_writers) > 0 and\
               self.recurring_bool(iteration_info, self.tensorboard_every):
                self.iteration_info.write_to_tensorboard(self.summary_writers)
            # save state to file
            if self.checkpoint_folder is not None\
               and self.recurring_bool(iteration_info, self.checkpoint_every):
                logger.log("saving checkpoint to %s, batches_seen: %i" %
                    (self.checkpoint_folder,
                     iteration_info.iterator_info.batches_seen))
                trainer.save_state(self.checkpoint_folder)
            # update progress bar
            self.pbar.update()

    def recurring_bool(self, iteration_info, every):
        return (iteration_info.iterator_info.batches_seen
                % every) == 0

    def enter(self, *args, **kwargs):
        if log_bool():
            self.pbar.enter(*args, **kwargs)

    def close(self):
        if not log_bool():
            return
        for writer in self.summary_writers.values():
            writer.close()
        if log_bool():
            self.pbar.exit()

    def save(self):
        for writer in self.summary_writers.values():
            writer.flush()

class ModelWrapper(nn.Module):
    def __init__(self, model, keys):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.keys = keys

    def forward(self, *values):
        kwargs = {k:v for k,v in zip(self.keys,values)}
        return tuple(self.model(**kwargs).values())
