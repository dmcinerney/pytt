#TODO: re-enforce the line character limit
import os
import subprocess
from threading import Thread
import tempfile
import zipfile
import socket
from datetime import datetime
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from pytt.utils import read_pickle, write_pickle
from pytt.distributed import collect_obj_on_rank0, log_bool
from pytt.progress_bar import ProgressBar
from pytt.logger import logger
from pytt.email import EmailSender

class Tracker:
    """
    Tracker object that creates a list (history) where each element is the info
    from one training iteration, handling info objects distributed across
    multiple devices.  Contains saving and loading functionality for use during
    checkpoint.  Also contains a string function which can be used for logging
    an iteration during training.
    """
    def __init__(self, pbar=None, print_every=1, checkpoint_every=1,
                 copy_checkpoint_every=None, checkpoint_folder=None,
                 tensorboard_every=1, summary_writers=['train', 'val'],
                 needs_graph=True, purge_step=None, email_every=None,
                 email_sender=None):
        self.print_every = print_every
        self.iteration_info = None
        if not log_bool():
            self.needs_graph = needs_graph
            return
        self.pbar = pbar if pbar is not None else ProgressBar()
        self.checkpoint_every = checkpoint_every
        self.copy_checkpoint_every = copy_checkpoint_every
        self.checkpoint_folder = checkpoint_folder
        if self.copy_checkpoint_every is not None:
            self.saved_checkpoints = 0
            subprocess.run(["mkdir", os.path.join(self.checkpoint_folder, "saved_checkpoints")])
        # set up tensorboard
        self.tensorboard_every = tensorboard_every
        if checkpoint_folder is None:
            datetime_machine = datetime.now().strftime('%b%d_%H-%M-%S')\
                               + '_' + socket.gethostname()
            self.tensorboard_folder = os.path.join('runs', datetime_machine)
        else:
            self.tensorboard_folder = os.path.join(checkpoint_folder, 'tensorboard')
        self.summary_writers = {k:
            SummaryWriter(log_dir=self.tensorboard_folder+'/'+k,
                          purge_step=purge_step)
            for k in summary_writers}
        self.needs_graph = needs_graph
        # set up email
        self.email_every = email_every
        if self.email_every is not None and log_bool():
            if email_sender is None:
                raise Exception
            self.email_sender = email_sender

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
            # copy checkpoint
            if self.copy_checkpoint_every is not None\
               and self.recurring_bool(iteration_info, self.copy_checkpoint_every):
                logger.log("copying to checkpoint number %i, batches_seen: %i" %
                    (self.saved_checkpoints,
                     iteration_info.iterator_info.batches_seen))
                self.copy_checkpoint_in_thread()
                logger.log("continuing")
            # email
            if self.recurring_bool(iteration_info, self.email_every):
                logger.log("sending email to %s, batches_seen: %i" %
                    (self.email_sender.receiver_email,
                     iteration_info.iterator_info.batches_seen))
                attachments = [] if len(self.summary_writers) <= 0 else\
                              create_tensorboard_attachment_generator(
                                  self.tensorboard_folder)
                self.email_sender(str(iteration_info),
                    attachments=attachments,
                    onfinish="Done sending email at %i batches_seen" %
                             iteration_info.iterator_info.batches_seen,
                    onerror="Error sending email at %i batches_seen!" %
                            iteration_info.iterator_info.batches_seen)
                logger.log("continuing")
            # update progress bar
            self.pbar.update()

    def recurring_bool(self, iteration_info, every):
        return every is not None and\
               (iteration_info.iterator_info.batches_seen
                % every) == 0\
               or iteration_info.iterator_info.batches_seen\
                  == iteration_info.iterator_info.total_batches

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

    def copy_checkpoint_in_thread(self):
        onfinish = "done copying to checkpoint number %i" % self.saved_checkpoints
        thread = Thread(target=copy_checkpoint, args=[self.checkpoint_folder, self.saved_checkpoints, onfinish])
        thread.start()
        self.saved_checkpoints += 1

class ModelWrapper(nn.Module):
    def __init__(self, model, keys):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.keys = keys

    def forward(self, *values):
        kwargs = {k:v for k,v in zip(self.keys,values)}
        return tuple(self.model(**kwargs).values())

# taken from https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory-in-python
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def create_tensorboard_attachment_generator(dir):
    basename = os.path.basename(dir)

    # create zipfile
    zf = tempfile.TemporaryFile(prefix=basename, suffix='.zip')
    zip = zipfile.ZipFile(zf, 'w')
    zipdir(dir, zip)
    zip.close()
    zf.seek(0)
    yield basename, basename + ".zip", zf
    zf.close()

def copy_checkpoint(checkpoint, checkpoint_num, onfinish="done copying checkpoint"):
    stuff = set(os.listdir(checkpoint))
    stuff.remove('saved_checkpoints')
    saved_checkpoint = os.path.join(checkpoint, 'saved_checkpoints', 'checkpoint%i' % checkpoint_num)
    subprocess.run(["mkdir", saved_checkpoint])
    for x in stuff:
        subprocess.run(["cp", "-r", os.path.join(checkpoint, x), saved_checkpoint])
    logger.log(onfinish)
