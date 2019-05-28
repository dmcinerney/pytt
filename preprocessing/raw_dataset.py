# handles loading from csv file using pandas
import pandas as pd

class RawDataset:
    """
    Acts as a handle for a dataset in a csv file, using pandas to read it
    """
    def __init__(self, filename):
        """
        Reads in dataframe from csv using pandas
        """
        self.df = pd.read_csv(filename)

    def __getitem__(self, index):
        """
        Returns the element at index index in the dataset
        """
        return dict(self.df.iloc[index])
        
    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.df)
