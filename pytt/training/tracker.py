import os
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from pytt.utils import read_pickle, write_pickle
from pytt.distributed import collect_obj_on_rank0, log_bool
import socket
from datetime import datetime

class Tracker:
    """
    Tracker object that creates a list (history) where each element is the info
    from one training iteration, handling info objects distributed across
    multiple devices.  Contains saving and loading functionality for use during
    checkpoint.  Also contains a string function which can be used for logging
    an iteration during training.
    """
    def __init__(self, checkpoint_folder=None, purge_step=None,
                 summary_writers=['train', 'val'], needs_graph=True):
        self.iteration_info = None
        if not log_bool():
            self.needs_graph = needs_graph
            return
        if checkpoint_folder is None:
            datetime_machine = datetime.now().strftime('%b%d_%H-%M-%S')\
                               + '_' + socket.gethostname()
            checkpoint_folder = os.path.join('runs', datetime_machine)
        self.summary_writers = {k:
            SummaryWriter(log_dir=checkpoint_folder+'/'+k,
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

    def register_iteration(self, iteration_info):
        self.iteration_info = iteration_info
        if dist.is_initialized():
            collected = collect_obj_on_rank0(
                self.iteration_info,
                ranks=self.iteration_info.iterator_info.subbatches.get_ranks())
            if collected is not None:
                self.iteration_info = sum(collected)
            else:
                self.iteration_info = None
        if log_bool() and len(self.summary_writers) > 0:
            self.iteration_info.write_to_tensorboard(self.summary_writers)

    def __str__(self):
        return str(self.iteration_info)

    def close(self):
        if not log_bool():
            return
        for writer in self.summary_writers.values():
            writer.close()

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
