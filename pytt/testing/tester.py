import torch
import torch.distributed as dist
from pytt.logger import logger
from pytt.distributed import collect_obj_on_rank0, log_bool
from pytt.iteration_info import BatchInfo
from pytt.utils import indent
from pytt.progress_bar import ProgressBar

class Tester:
    def __init__(self, model, batch_iterator, batch_info_class=BatchInfo, pbar=None, print_every=1):
        self.model = model
        self.batch_iterator = batch_iterator
        self.pbar = pbar if pbar is not None else ProgressBar()
        self.batch_info_class = batch_info_class
        self.current_batch_info = 0
        self.total_batch_info = 0
        self.print_every = print_every

    def test(self, loss_func, statistics_func=None, use_pbar=True):
        if use_pbar:
            self.pbar.enter(len(self.batch_iterator.indices_iterator))
        try:
            while True:
                self.register_iteration(
                    self.process_batch(self.next_batch(), loss_func, statistics_func=statistics_func))
        except StopIteration:
            if use_pbar:
                self.pbar.exit()
        return self.total_batch_info

    def process_batch(self, batch, loss_func, statistics_func=None):
        with torch.autograd.no_grad():
            # run batch through the model
            outputs = self.model(**batch.get_observed())
            # calculate loss using the outputs of the model
            loss = loss_func(**outputs, **batch.get_target())
            # if statistics function is given, calculate it
            if statistics_func is not None:
                stats = statistics_func(**outputs, **batch.get_target())
        step_dict = {"loss":loss, "_batch_length":torch.tensor(len(batch))}
        if statistics_func is not None:
            step_dict.update(stats)
        return step_dict

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
                if self.batch_iterator.iterator_info().batches_seen % self.print_every == 0:
                    logger.log(self.get_log_string())
            self.current_batch_info = 0

    def get_log_string(self):
        log_string = ""
        log_string += str(self.batch_iterator.iterator_info())
        log_string += "\n  Running Stats:\n"
        log_string += indent(str(self.total_batch_info), "    ")
        return log_string

    def next_batch(self):
        batch = next(self.batch_iterator)
        if self.batch_iterator.take_step():
            self.pbar.update()
        return batch
