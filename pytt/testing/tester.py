import torch
import torch.distributed as dist
from pytt.logger import logger
from pytt.distributed import collect_obj_on_rank0, log_bool
from pytt.iteration_info import BatchInfo
from pytt.utils import indent
from pytt.progress_bar import ProgressBar
from pytt.testing.datapoint_processor import DatapointProcessor

class Tester(DatapointProcessor):
    """
    Tester object containing model and a batch_iterator. It can use a custom
    batch info class and progress_bar class, and the frequency of printing can
    be controlled.
    """
    def __init__(self, model, batch_iterator, batch_info_class=BatchInfo,
                 pbar=None, print_every=1):
        self.model = model
        self.batch_iterator = batch_iterator
        self.pbar = pbar if pbar is not None else ProgressBar()
        self.batch_info_class = batch_info_class
        self.current_batch_info = 0
        self.total_batch_info = 0
        self.print_every = print_every

    def test(self, test_func, use_pbar=True):
        """
        Tests the model by calling next on the batch_iterator until it throws a
        StopIteration Exception.  It uses a test_func to process the output of
        the model and return any relevant batch statistics to accumulate
        throughout testing. The user should use this function to do any writing
        of individual instance outputs to disk. The function should only return
        a dictionary of floats and not any other data type. Statistics returned
        by the test_func will be accumulated in self.total_batch_info as a
        batch_info object. The running statistics will also be logged. The
        progress bar is used unless use_pbar is specified False.
        """
        if use_pbar:
            self.pbar.enter(total=len(self.batch_iterator.indices_iterator),
                initial=self.batch_iterator.iterator_info().batches_seen)
        try:
            while True:
                self.register_iteration(
                    self.process_batch(next(self.batch_iterator), test_func))
                if self.batch_iterator.take_step():
                    self.pbar.update()
        except StopIteration:
            if use_pbar:
                self.pbar.exit()
        return self.total_batch_info

    def register_iteration(self, batch_info):
        self.current_batch_info += self.batch_info_class({
            k:v.item() for k,v in batch_info.items()})
        if self.batch_iterator.take_step():
            if dist.is_initialized():
                collected = collect_obj_on_rank0(self.current_batch_info)
                if collected is not None:
                    self.current_batch_info = sum(collected)
            if log_bool():
                self.total_batch_info += self.current_batch_info
                if self.batch_iterator.iterator_info().batches_seen\
                   % self.print_every == 0:
                    logger.log(self.get_log_string())
            self.current_batch_info = 0

    def get_log_string(self):
        log_string = ""
        log_string += str(self.batch_iterator.iterator_info())
        log_string += "\n  Running Stats:\n"
        log_string += indent(str(self.total_batch_info), "    ")
        return log_string
