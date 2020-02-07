import os
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from pytt.logger import logger
from pytt.distributed import collect_obj_on_rank0, log_bool
from pytt.utils import indent
from pytt.progress_bar import ProgressBar
from pytt.testing.datapoint_processor import DatapointProcessor

class Tester(DatapointProcessor):
    """
    Tester object containing model and a batch_iterator. It can use a custom
    batch info class and progress_bar class, and the frequency of printing can
    be controlled.
    """
    def __init__(self, model, postprocessor, batch_iterator, pbar=None,
                 print_every=1, tensorboard_every=1, tensorboard_dir=None):
        super(Tester, self).__init__(model, postprocessor)
        self.batch_iterator = batch_iterator
        self.pbar = pbar if pbar is not None else ProgressBar()
        self.current_output_batch = 0
        self.total_output_batch = 0
        self.print_every = print_every
        self.tensorboard_every = tensorboard_every
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

    def test(self, use_pbar=True):
        """
        Tests the model by calling next on the batch_iterator until it throws a
        StopIteration Exception.  It uses a test_func to process the output of
        the model and return any relevant batch statistics to accumulate
        throughout testing. The user should use this function to do any writing
        of individual instance outputs to disk. The function should only return
        a dictionary of floats and not any other data type. Statistics returned
        by the test_func will be accumulated in self.total_output_batch as an
        output_batch object. The running statistics will also be logged. The
        progress bar is used unless use_pbar is specified False.
        """
        if use_pbar:
            self.pbar.enter(total=len(self.batch_iterator.indices_iterator),
                initial=self.batch_iterator.iterator_info().batches_seen)
        try:
            while True:
                output_batch = self.process_batch(
                    next(self.batch_iterator))
                self.register_iteration(output_batch)
                if self.batch_iterator.take_step():
                    self.pbar.update()
        except StopIteration:
            if use_pbar:
                self.pbar.exit()
        return self.total_output_batch

    def register_iteration(self, output_batch):
        self.current_output_batch += output_batch
        if self.batch_iterator.take_step():
            if dist.is_initialized():
                collected = collect_obj_on_rank0(self.current_output_batch)
                if collected is not None:
                    self.current_output_batch = sum(collected)
            if log_bool():
                self.total_output_batch += self.current_output_batch
                if self.batch_iterator.iterator_info().batches_seen\
                   % self.print_every == 0:
                    logger.log(self.get_log_string())
                if self.batch_iterator.iterator_info().batches_seen\
                   % self.tensorboard_every == 0:
                    self.current_output_batch.write_to_tensorboard(
                        self.writer, self.batch_iterator.iterator_info())
            self.current_output_batch = 0

    def get_log_string(self):
        log_string = ""
        log_string += str(self.batch_iterator.iterator_info())
        log_string += "\n  Running Stats:\n"
        log_string += indent(str(self.total_output_batch), "    ")
        return log_string
