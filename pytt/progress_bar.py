from tqdm import tqdm
from pytt.distributed import log_bool
from pytt.logger import logger

class ProgressBar:
    def __init__(self):
        self.pbar = None

    def enter(self, *args, **kwargs):
        if log_bool():
            self.pbar = self.init_pbar(*args, **kwargs)
        logger.add_progress_bar(tqdm)

    def exit(self):
        if log_bool():
            self.pbar.close()
            self.pbar = None
        logger.remove_progress_bar()

    def update(self, n=1):
        if self.pbar is not None:
            self.pbar.update(n=n)

    def init_pbar(self, *args, mininterval=1, leave=len(logger.pbars) == 0, **kwargs):
        return tqdm(*args, mininterval=mininterval, leave=leave, **kwargs)
