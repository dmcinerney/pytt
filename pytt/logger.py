# contains all logging operations
import torch.distributed as dist

class LoggerClass:
    def __init__(self):
        self.verbosity = 1
        self.pbars = []

    def log(self, obj, verbosity=1):
        if verbosity <= self.verbosity:
            self.print_func(obj)

    def print_func(self, obj):
        if len(self.pbars) == 0:
            print(obj)
        else:
            self.pbars[-1].write(obj)

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity

    def add_progress_bar(self, pbar):
        self.pbars.append(pbar)

    def remove_progress_bar(self):
        self.pbars.pop()

logger = LoggerClass()
