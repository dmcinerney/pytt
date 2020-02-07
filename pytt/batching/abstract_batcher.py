import pickle as pkl
import torch
from torch.utils.data import Sampler
from pytt.utils import split, read_pickle, write_pickle

class AbstractBatcher:
    """
    Handles all conversions from raw data to a batch and vice versa

    This is an abstract class that allows a very flexible framework for creating
    batchers.  See standard_batchers.py for examples of standard implementations
    of this abstract architecture.
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
        Returns a batch made from the raw datapoints at the given indices in the
        dataset
        """
        raw_datapoint_generator = (dataset[i] for i in indices)
        return self.batch_from_raw(raw_datapoint_generator)

    def batch_from_raw(self, raw_datapoints):
        """
        Returns a batch made from the raw datapoints given
        """
        processed_datapoint_generator = (self.process_datapoint(raw_datapoint)
                                         for raw_datapoint in raw_datapoints)
        return self.batch(processed_datapoint_generator)

    def batch(self, instances):
        """
        Returns a batch made from instance objects
        """
        raise NotImplementedError

    def batch_iterator(self, dataset, indices_iterator):
        """
        Returns an iterator of batches from the dataset iterating according to
        the input indices_iterator
        """
        raise NotImplementedError


class AbstractIndicesIterator(Sampler):
    """
    Handles iterating of indices of the dataset

    This is an abstract class that allows a very flexible framework for creating
    indices iterators.  See standard_batchers.py for examples of standard
    implementations of this abstract architecture.
    """
    @staticmethod
    def load(filename):
        """
        Loads an IndicesIterator from file using pickle
        """
        return read_pickle(filename)

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
        Returns an IteratorInfo object
        """
        raise NotImplementedError

    def save(self, filename):
        """
        Saves an IndicesIterator to file using pickle
        """
        write_pickle(self, filename)


class AbstractBatchIterator:
    """
    Iterates over a dataset in batches

    This is an abstract class that allows a very flexible framework for creating
    batch iterators.  See standard_batchers.py for examples of standard
    implementations of this abstract architecture.
    """
    def __init__(self, batcher, dataset, indices_iterator):
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
        Returns an IteratorInfo object
        """
        raise NotImplementedError


class IteratorInfo:
    """
    Contains any necessary info concerning the current state of the iterator
    """
    def __init__(self, batches_seen, samples_seen):
        self.batches_seen = batches_seen
        self.samples_seen = samples_seen

    def __str__(self):
        return "batches_seen: "+str(self.batches_seen)\
             + ", samples_seen: "+str(self.samples_seen)


class AbstractInstance:
    """
    Handles preprocessing of raw datapoint, holding all information needed on a
    datapoint

    This is an abstract class that allows a very flexible framework for creating
    instances.  See standard_batchers.py for examples of standard
    implementations of this abstract architecture.
    """
    def __init__(self, raw_datapoint):
        """
        Initializes the instance object using a raw data dictionary

        inputs:
            1) raw datapoint

        IMPORTANT NOTE: this needs to create three object attributes:
            1) self.raw_datapoint - a reference to the raw datapoint
            2) self.datapoint - a processed dictionary of anything needed at the
                batch level
            3) self.observed - a subset of keys of the self.datapoint object
                specifying what should be observed by the model (the rest are
                assumed to be unobserved but needed during post-processing)
        """
        raise NotImplementedError

    def keep_in_batch(self):
        """
        Returns an object containing anything that might be needed on the batch
        level (defaults to returning None)
        """
        return None

    def to(self, device):
        """
        Moves Instance input tensors to the specified device
        """
        for k,v in self.datapoint.items():
            if isinstance(v, torch.Tensor):
                self.datapoint[k] = v.to(device=device)
        return self


class AbstractBatch:
    """
    Handles collecting instances into a batch

    This is an abstract class that allows a very flexible framework for creating
    batches.  See standard_batchers.py for examples of standard implementations
    of this abstract architecture.
    """
    # TODO: determine if this function should be deleted
    @classmethod
    def init_batches_across_devices(cls, instances, devices):
        """
        Returns a list of batches of approximately equal size that cover all the
        instances in order, each batch corresponding to one of the devices
        specified
        """
        batches = []
        offset = 0
        for device,batch_size in zip(devices,
                                     split(len(instances),len(devices))):
            batches.append(cls(instances[offset:offset+batch_size]).to(device))
            offset += batch_size
        return batches

    @staticmethod
    def collate_datapoints(datapoints):
        """
        Returns a dictionary that is the collated version of the list of datapoints
        given
        """
        raise NotImplementedError

    def __init__(self, instances):
        """
        Initializes the batch object using a list of instance objects

        inputs:
            1) a list of instances

        IMPORTANT NOTE: this needs to create two object attributes:
            1) self.instances - a list of references to the objects for each
                instance output from instance.keep_in_batch()
            1) self.collated_datapoints - a dictionary that is the collated
                version of self.datapoints
            2) self.observed - same as in instance
        """
        self.instances = [instance.keep_in_batch() for instance in instances]
        self.collated_datapoints = self.__class__.collate_datapoints(
            [instance.datapoint for instance in instances])
        self.observed = instances[0].observed

    def __len__(self):
        """
        Returns the number of datapoints in the batch
        """
        return len(self.instances)

    def get_observed(self):
        """
        Return the objects needed for the forward pass of the model
        """
        return {k:self.collated_datapoints[k] for k in self.observed}

    def get_target(self):
        """
        Return the objects needed for the loss and statistics functions
        """
        return {k:self.collated_datapoints[k]
            for k in self.collated_datapoints.keys() if k not in self.observed}

    def to(self, device):
        """
        Moves the batch to the specified device
        """
        for k,v in self.collated_datapoints.items():
            if isinstance(v, torch.Tensor):
                self.collated_datapoints[k] = v.to(device=device)
        return self
