import torch
import torch.distributed as dist
from pytt.logger import logger
from pytt.distributed import collect_obj_on_rank0
from pytt.iteration_info import BatchInfo

class Tester:
    def __init__(self, model, batch_iterator, batch_info_class=BatchInfo):
        self.model = model
        self.batch_iterator = batch_iterator
        self.batch_info_class = batch_info_class
        self.current_batch_info = 0
        self.total_batch_info = 0

    def test(self, loss_func, statistics_func=None):
        for batch in self.batch_iterator:
            self.register_iteration(
                self.process_batch(batch, loss_func, statistics_func=statistics_func))
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
            if not dist.is_initialized()\
               or dist.is_initialized() and dist.get_rank() == 0:
                self.total_batch_info += self.current_batch_info
                logger.log(self.total_batch_info)
            self.current_batch_info = 0
