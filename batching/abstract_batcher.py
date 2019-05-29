from torch.utils.data import Sampler
import pickle as pkl
from utils import split

class AbstractBatcher:
    """
    Handles all conversions from raw data to a batch and vice versa

    This is an abstract class that allows a very flexible framework for creating batchers.
    See standard_batchers.py for examples of standard implementations of this abstract architecture.
    """
    def __init__(self):
        raise NotImplementedError

    def process_datapoint(self, raw_datapoint):
        """
        Returns an instance made from the raw datapoint
        """
        raise NotImplementedError

    def batch_from_dataset(self, dataset, indices):
        """
        Returns a batch made from the raw datapoints at the given indices in the dataset
        """
        raw_datapoint_generator = (dataset[i] for i in indices)
        return self.batch_from_raw(raw_datapoint_generator)

    def batch_from_raw(self, raw_datapoints):
        """
        Returns a batch made from the raw datapoints given
        """
        processed_datapoint_generator = (self.process_datapoint(raw_datapoint) for raw_datapoint in raw_datapoints)
        return self.batch(processed_datapoint_generator)

    def batch(self, instances):
        """
        Returns a batch made from instance objects
        """
        raise NotImplementedError

    def out_batch_to_readable(self, output_batch):
        """
        Returns a list of readable outputs given the output batch from the model
        """
        raise NotImplementedError

    def batch_iterator(self, dataset, batch_size):
        """
        Returns an iterator of batches from the dataset
        """
        raise NotImplementedError


class AbstractBatchIterator:
    """
    Iterates over a dataset in batches

    This is an abstract class that allows a very flexible framework for creating batch iterators.
    See standard_batchers.py for examples of standard implementations of this abstract architecture.
    """
    def __init__(self, batcher, dataset, batch_size):
        """
        Should initialize any iterators needed
        """
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next batch object
        """
        raise NotImplementedError

    def iterator_info(self):
        """
        Returns a dictionary describing what has been already iterated through
        """
        raise NotImplementedError

class AbstractInstance:
    """
    Handles preprocessing of raw datapoint, holding all information needed on a datapoint

    This is an abstract class that allows a very flexible framework for creating instances.
    See standard_batchers.py for examples of standard implementations of this abstract architecture.
    """
    def __init__(self, raw_datapoint):
        """
        Initializes the instance object using a raw data dictionary

        inputs:
            1) raw datapoint

        IMPORTANT NOTE: this needs to create two object attributes:
            1) self.datapoint - a reference to the raw datapoint and any other information
                that it is necessary to keep around
            2) self.input - a processed dictionary of tensors
        """
        raise NotImplementedError

    def to(self, device):
        """
        Moves Instance input tensors to the specified device
        """
        for k,v in self.input.items():
            self.input[k] = v.to(device=device)
        return self

class AbstractBatch:
    """
    Handles collecting instances into a batch

    This is an abstract class that allows a very flexible framework for creating batches.
    See standard_batchers.py for examples of standard implementations of this abstract architecture.
    """
    @classmethod
    def init_batches_across_devices(cls, instances, devices):
        """
        Returns a list of batches of approximately equal size that cover all the
        instances in order, each batch corresponding to one of the devices specified
        """
        batches = []
        offset = 0
        for device,batch_size in zip(devices, split(len(instances), len(devices))):
            batches.append(cls(instances[offset:offset+batch_size]).to(device))
            offset += batch_size
        return batches

    def __init__(self, instances):
        """
        Initializes the batch object using a list of instance objects

        inputs:
            1) list of instances

        IMPORTANT NOTE: this needs to create two object attributes:
            1) self.datapoints - a reference to the raw datapoints and any other information
                that it is necessary to keep around
            2) self.inputs - a processed dictionary of tensors
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the number of datapoints in the batch
        """
        return len(self.datapoints)

    def to(self, device):
        """
        Moves the batch to the specified device
        """
        for k,v in self.inputs.items():
            self.inputs[k] = v.to(device=device)
        return self

class AbstractOutputInstance:
    """
    Handles postprocessing of model ouptuts into something readable

    This is an abstract class that allows a very flexible framework for creating output instances.
    See standard_batchers.py for examples of standard implementations of this abstract architecture.
    """
    def __init__(self, output):
        """
        Initializes the output instance with the output for a single example
        """
        raise NotImplementedError


class AbstractOutputBatch:
    """
    Handles collecting model outputs into an output_batch object

    This is an abstract class that allows a very flexible framework for creating output batches.
    See standard_batchers.py for examples of standard implementations of this abstract architecture.
    """
    @staticmethod
    def combine(output_batches):
        """
        Combines multiple AbstractOutputBatch objects into one
        """
        raise NotImplementedError

    def __init__(self, model_output):
        """
        Initializes output batch object from a raw model output
        """
        raise NotImplementedError

    def get_output_insances(self):
        """
        Returns a list of AbstractOutputInstance objects
        """
        raise NotImplementedError



# possible helper for batch iterator
class AbstractIndicesIterator(Sampler):
    """
    Handles iterating of indices of the dataset

    This is an abstract class that allows a very flexible framework for creating indices iterators.
    See standard_batchers.py for examples of standard implementations of this abstract architecture.
    """
    @staticmethod
    def load(filename):
        """
        Loads an IndicesIterator from file using pickle
        """
        with open(filename, 'rb') as f:
            return pkl.load(f)

    def __init__(self, source_length):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next batch indices
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the length of the iterator
        """
        raise NotImplementedError


    def iterator_info(self):
        """
        Returns a dictionary describing what has been already iterated through
        """
        raise NotImplementedError

    def save(self, filename):
        """
        Saves an IndicesIterator to file using pickle
        """
        with open(filename, 'wb') as f:
            pkl.dump(self, f)