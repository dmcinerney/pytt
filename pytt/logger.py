# contains all logging operations
import torch.distributed as dist

class LoggerClass:
    def __init__(self):
        self.verbosity = 1
        self.pbar = None

    def log(self, obj, verbosity=1):
        if verbosity <= self.verbosity:
            self.print_func(obj)

    def print_func(self, obj):
        if self.pbar is None:
            print(obj)
        else:
            self.pbar.write(obj)

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity

    def set_progress_bar(self, pbar):
        self.pbar = pbar

logger = LoggerClass()
