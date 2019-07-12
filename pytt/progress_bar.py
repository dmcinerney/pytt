from tqdm import tqdm
from pytt.distributed import log_bool
from pytt.logger import logger

class ProgressBar:
    def __init__(self):
        self.pbar = None

    def enter(self, total):
        if log_bool():
            self.pbar = self.init_pbar(total)
        logger.add_progress_bar(tqdm)

    def exit(self):
        if log_bool():
            self.pbar.close()
            self.pbar = None
        logger.remove_progress_bar()

    def update(self):
        if self.pbar is not None:
            self.pbar.update()

    def init_pbar(self, total):
        return tqdm(total=total, mininterval=1, leave=len(logger.pbars) == 0)
