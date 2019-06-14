# contains all logging operations
import torch.distributed as dist

class Logger:
    @classmethod
    def log(cls, obj):
        print(obj)

class TrainLogger(Logger):
    pass
