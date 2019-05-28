from abstract_batcher import AbstractBatcher,
                             AbstractBatchIterator,
                             AbstractInstance,
                             AbstractBatch,
                             AbstractIndicesIterator
import torch
from torch.utils.data import Dataset,
                             DataLoader,
                             Sampler,
                             SequentialSampler,
                             RandomSampler,
                             BatchSampler
import copy

class StandardBatcher(AbstractBatcher):
    """
    Implementation of a simple AbstractBatcher which does no processing (see StandardInstance) and returns a StandardBatchIterator
    from the batch_iterator function

    All functions without comments are described in the superclass.
    """
    def __init__(self):
        pass

    def process_datapoint(self, raw_datapoint):
        return StandardInstance(raw_datapoint)

    def batch_from_raw(self, raw_datapoints):
        return StandardBatch([process_raw(raw_datapoint) for raw_datapoint in raw_datapoints])

    def out_batch_to_readable(self, output_batch):
        # TODO: implement this
        raise NotImplementedError

    def batch_iterator(self, dataset, batch_size, indices_iterator=None, num_workers=0, random=False, epochs=None, iterations=None):
        return StandardBatchIterator(
            self,
            dataset,
            batch_size,
            indices_iterator=indices_iterator,
            num_workers=num_workers,
            random=random,
            epochs=epochs,
            iterations=iterations
        )

class StandardBatchIterator(AbstractBatchIterator):
    """
    Implementation of a simple BatchIterator which allows one to specify various types of iterators over the dataset:
        if random:
            create and use a random indices_iterator (see RandomIndicesIterator)
        else:
            create and use a sequential indices_iterator (see SequentialIndicesIterator)
    It also gives an option to give an indices_iterator which should be an instance of an AbstractIndicesIterator

    All functions without comments are described in the superclass.
    """
    def __init__(self, batcher, dataset, batch_size, indices_iterator=None, num_workers=0, random=None, epochs=None, iterations=None):
        dataset = DatasetWrapper(dataset, batcher)
        if indices_iterator is not None:
            if random is not None or epochs is not None or iterations is not None:
                raise Exception
            self.indices_iterator = indices_iterator
        else:
            if random:
                self.indices_iterator = RandomIndicesIterator(len(dataset), batch_size, epochs=epochs, iterations=iterations)
            else:
                self.indices_iterator = SequentialIndicesIterator(len(dataset), batch_size)
        # TODO: change back to not an attribute
        self.indices_iterator_copy = copy.deepcopy(self.indices_iterator)
        self.dataloaderiter = iter(DataLoader(
            dataset,
            batch_sampler=self.indices_iterator_copy,
            collate_fn=lambda instances: StandardBatch(instances), # TODO: make sure this works
            num_workers=num_workers
        ))

    def __iter__(self):
        return self

    def __next__(self):
        # TODO: test this function and then remove the breakpoint
        import pdb; pdb.set_trace()
        next(self.indices_iterator)
        return next(self.dataloaderiter)

    def iterator_info(self):
        return self.indices_iterator.iterator_info()

class DatasetWrapper(Dataset):
    """
    Implementation of a Dataset that uses the input batcher to process each datapoint
    """
    def __init__(self, dataset, batcher):
        self.dataset = dataset
        self.batcher = batcher

    def __getitem__(self, index):
        """
        Returns the processed datapoint (an Instance) at index index using the batcher
        """
        return self.batcher.process_datapoint(self.dataset[index])

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.dataset)

class SequentialIndicesIterator(AbstractIndicesIterator):
    """
    Implementation of an AbstractIndicesIterator that iterates over the dataset in order

    All functions without comments are described in the superclass.
    """
    def __init__(self, source_length, batch_size):
        self.samples_seen = 0
        self.batches_seen = 0
        self.sample_iter = iter(SequentialSampler(range(source_length)))
        self.set_batch_iter(batch_size)

    def __next__(self):
        indices = next(self.sample_iter)
        self.samples_seen += len(indices)
        self.batches_seen += 1
        return indices

    def iterator_info(self):
        return {"batches_seen":self.batches_seen, "samples_seen":self.samples_seen}

    def set_batch_iter(batch_size):
        """
        Sets the batch iterator to a new batch iterator with batch size batch_size
        """
        self.batch_size = batch_size
        self.batch_iter = iter(BatchSampler(IteratorWrapper(self.sample_iter), batch_size, False))

class RandomIndicesIterator(AbstractIndicesIterator):
    """
    Implementation of an AbstractIndicesIterator which allows one to specify various types of random samplers:
        if epochs is None and iterations is None:
            the object is an infinite random sampler with replacement
        elif epochs is None and isinstance(iterations, int):
            the object is a random sampler with replacement that stops after iterations iterations
        elif isinstance(epochs, bool) and epochs == True and isinstance(iterations, int):
            the object is a random sampler without replacement that stops after iterations iterations and counts epochs
        elif isinstance(epochs, int) and iterations is None:
            the object is a random sampler without replacement that stops after epochs epochs and counts iterations
        else:
            raise Exception

    All functions without comments are described in the superclass.
    """
    def __init__(self, source_length, batch_size, epochs=None, iterations=None):
        self.samples_seen = 0
        self.batches_seen = 0
        self.epochs_seen = 0
        self.num_epochs = epochs
        self.num_iterations = iterations
        self.replacement = self.check_flags_return_if_replacement(epochs, iterations)
        self.sampler = RandomSampler(range(source_length), replacement=self.replacement)
        self.sample_iter = iter(self.sampler)
        self.set_batch_iter(batch_size)

    def __next__(self):
        try:
            indices = next(self.batch_iter)
        except StopIteration:
            self.epochs_seen += 1
            if isinstance(self.num_epochs, int) and self.epochs_seen+1 >= self.num_epochs:
                raise StopIteration
            self.sample_iter = iter(self.sampler)
            self.set_batch_iter(self.batch_size)
            indices = next(self.batch_iter)
        if isinstance(self.num_iterations, int) and self.batches_seen >= self.num_iterations:
            raise StopIteration
        self.samples_seen += len(indices)
        self.batches_seen += 1
        return indices

    def iterator_info(self):
        return {"batches_seen":self.batches_seen, "samples_seen":self.samples_seen, "epochs_seen":self.epochs_seen}

    def set_batch_iter(batch_size):
        """
        Sets the batch iterator to a new batch iterator with batch size batch_size
        """
        self.batch_size = batch_size
        self.batch_iter = iter(BatchSampler(IteratorWrapper(self.sample_iter), batch_size, False))

    def set_stop(epochs, iterations):
        """
        Changes the stopping point of the iterator by respecifying the epochs and iterations parameters

        This is only allowed if the epochs and iterations are still compatable with the current sampler,
        meaning they don't change a with replacement to a without replacement sampler or vice versa.
        """
        if self.check_flags_return_if_replacement(epochs, iterations) != self.replacement:
            raise Exception
        self.num_epochs = epochs
        self.num_iterations = iterations

    def check_flags_return_if_replacement(self, epochs, iterations):
        """
        Checks if the flags are compatable and returns if the random sampling is done with replacement
        """
        if epochs is None and iterations is None or\
           epochs is None and isinstance(iterations, int):
            return True
        elif isinstance(epochs, bool) and epochs == True and isinstance(iterations, int) or\
             isinstance(epochs, int) and iterations is None:
            return False
        else:
            raise Exception

class IteratorWrapper(Sampler):
    """
    Small wrapper so that a Sampler's iterator can act as a Sampler itself,
    allowing it to be wrapped in a BatchSampler (used in IndicesIterators)
    """
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return iter(self.iterator)

    def __len__(self):
        return len(self.iterator)

def StandardInstance(AbstractInstance):
    """
    Implementation of an AbstractInstance that does no processing

    All functions without comments are described in the superclass.
    """
    def __init__(self, raw_datapoint):
        self.raw_datapoint = raw_datapoint
        self.processed_datapoint = raw_datapoint

def StandardBatch(AbstractBatch):
    """
    Implementation of a simple AbstractBatch

    All functions without comments are described in the superclass.
    """
    def __init__(self, instances):
        self.raw_datapoints = [instance.raw_datapoint for instance in instances]
        self.processed_datapoints = {k:pad_and_concat((instance.processed_datapoint[k] for instance in instances))
                                     for k in instance[0].processed_datapoints.keys()}

    def split(self, n):
        # TODO: implement this
        raise NotImplementedError

    def __len__(self):
        return len(self.raw_datapoints)

    def to(self, device):
        # TODO: implement this
        raise NotImplementedError

# TODO: figure out if something can be done for the output batch and output instance objects

# helper functions used to create a StandardBatch object

def get_max_dims(tensors):
    """
    Returns None if the tensors are all the same size
    and the maximum size in each dimension otherwise
    """
    dim = tensors[0].dim()
    max_size = [0]*dim
    different = False
    for tensor in tensors:
        if tensor.dim() != dim:
            raise Exception
        for i in range(dim):
            if not different:
                different = max_size[i] != tensor.size(i)
            max_size[i] = max(max_size[i], tensor.size(i))
    if different:
        return max_size
    else:
        return None

def pad_and_concat(tensors, max_size=None, auto=True):
    """
    Returns concatenated tensors with the added batch dimension being first
    """
    dim = tensors[0].dim()
    if auto:
        if max_size is not None:
            raise Exception
        max_size = get_max_dims(tensors)
    concatenated_tensor = []
    for tensor in tensors:
        if tensor.dim() != dim:
            raise Exception
        if max_size is not None:
            padding = []
            for i in range(dim-1,-1,-1):
                padding.extend([0,max_size[i]-tensor.size(i)])
            new_tensor = F.pad(tensor, tuple(padding))
        else:
            new_tensor = tensor
        concatenated_tensor.append(new_tensor.view(1,*new_tensor.size()))
    concatenated_tensor = torch.cat(concatenated_tensor, 0)
    return concatenated_tensor
