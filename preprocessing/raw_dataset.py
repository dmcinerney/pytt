# handles loading from pandas file
import pandas as pd

class RawDataset:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)