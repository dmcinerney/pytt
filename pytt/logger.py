# contains all logging operations
import torch.distributed as dist

class LoggerClass:
    def __init__(self):
        self.verbosity = 1

    def log(self, obj, verbosity=1):
        if verbosity <= self.verbosity:
            print(obj)

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity

logger = LoggerClass()
