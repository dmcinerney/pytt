import math
import copy
import queue
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset,\
                             DataLoader,\
                             Sampler,\
                             SequentialSampler,\
                             RandomSampler
from pytt.batching.abstract_batcher import AbstractBatcher,\
                                           AbstractBatchIterator,\
                                           AbstractInstance,\
                                           AbstractBatch,\
                                           AbstractIndicesIterator
from pytt.utils import split_range


class StandardBatcher(AbstractBatcher):
    """
    Implementation of a simple AbstractBatcher which does no processing (see
    StandardInstance) and returns a StandardBatchIterator from the
    batch_iterator function

    All functions without comments are described in the superclass.
    """
    def __init__(self):
        pass

    def process_datapoint(self, raw_datapoint):
        return StandardInstance(raw_datapoint)

    def batch_from_dataset(self, dataset, indices, devices=None):
        raw_datapoint_generator = (dataset[i] for i in indices)
        return self.batch_from_raw(raw_datapoint_generator, devices=devices)

    def batch_from_raw(self, raw_datapoints, devices=None):
        processed_datapoint_generator = (self.process_datapoint(raw_datapoint)
                                         for raw_datapoint in raw_datapoints)
        return self.batch(processed_datapoint_generator, devices=devices)

    def batch(self, instances, devices=None):
        if devices is None:
            return StandardBatch(list(instances))
        else:
            if isinstance(devices, str):
                return StandardBatch(list(instances)).to(devices)
            else:
                return StandardBatch.init_batches_across_devices(
                    list(instances), devices)

    def out_batch_to_readable(self, output_batch):
        # TODO: implement this
        raise NotImplementedError

    def batch_iterator(self, dataset, batch_size=None, subbatches=None,
                       random=None, epochs=None, iterations=None,
                       indices_iterator=None, num_workers=0, devices=None):
        """
        TODO: fill out this comment
        """
        return StandardBatchIterator(
            self,
            dataset,
            batch_size=batch_size,
            subbatches=subbatches,
            random=random,
            epochs=epochs,
            iterations=iterations,
            indices_iterator=indices_iterator,
            num_workers=num_workers,
            devices=devices,
        )


class StandardBatchIterator(AbstractBatchIterator):
    """
    Implementation of a simple BatchIterator which allows one to specify various
    types of iterators over the dataset:
        if random:
            create and use a random indices_iterator (see RandomIndicesIterator)
        else:
            create and use a sequential indices_iterator (see
            SequentialIndicesIterator)
    It also gives an option to give an indices_iterator which should be an
    instance of an AbstractIndicesIterator.

    MultiBatchIndicesIterator: This implementation also allows for batches to be
    split up and loaded to multiple devices and/or loaded sequentially with
    delayed updates.

    IteratorQueueWrapper: This implementation uses a pytorch DataLoader, but
    adds functionality such as allowing one to access an indices iterator
    corresponding to the state of the iterator when it loaded the indices for
    the current batch.  This is obtained by wrapping the indices iterator in a
    custom IteratorQueueWrapper, which saves the states of the iterator after
    each next call performed by the DataLoader on the batch_sampler until the
    corresponding next call to the StandardBatchIterator catches up.  The first
    iterator state in the queue is then popped off to set the indices iterator
    to the correct state (the state just after the current batch's indices were
    loaded.)

    All functions without comments are described in the superclass.
    """
    def __init__(self, batcher, dataset, batch_size=None, subbatches=None,
                 random=None, epochs=None, iterations=None,
                 indices_iterator=None, num_workers=0, devices=None):
        dataset = DatasetWrapper(dataset, batcher)
        if indices_iterator is not None:
            # Checks that all of the indices_iterator initializer arguments are
            #   None
            # NOTE: subbatches is not included in this because it is used to
            #   initialize the MultiBatchIndicesIterator, a wrapper, not the
            #   indices_iterator itself
            if batch_size is not None or random is not None\
               or epochs is not None or iterations is not None:
                raise Exception
        else:
            indices_iterator = self.init_indices_iterator(
                len(dataset),
                batch_size=batch_size,
                random=random,
                epochs=epochs,
                iterations=iterations
            )
        # Does nothing when not distributed or split into subbatches
        self.multi_batch_indices_iterator = MultiBatchIndicesIterator(
            indices_iterator, subbatches_per_process=subbatches)
        # Wraps the iterator in a queue to ensure the correct iterator state is
        #   kept for logging and checkpointing information
        # WARNING: this means that if multi_batch_indices_iterator is changed,
        #   the dataloader will not change behavior, one would need to
        #   reinitialize the BatchIterator
        self.queue_wrapped_indices_iterator = IteratorQueueWrapper(
            copy.deepcopy(self.multi_batch_indices_iterator))
        # A callable object that uses the batcher to collate instances into a
        #   batch
        collate_fn = CollateFnObject(batcher, devices)
        # Creating the DataLoader object, but only using its iterator (in order
        #   to start over, you need to reinitialize the BatchIterator)
        self.dataloaderiter = iter(DataLoader(
            dataset,
            batch_sampler=self.queue_wrapped_indices_iterator,
            collate_fn=collate_fn,
            num_workers=num_workers
        ))

    def init_indices_iterator(self, source_length, batch_size=None, random=None,
                              epochs=None, iterations=None):
        """
        Initializes some common types of indices_iterators (only called if one
        is not given)
        """
        # Checks that the batch_size is set because it is needed for both
        #   IndicesIterator types
        if batch_size is None:
            raise Exception
        if random:
            indices_iterator = RandomIndicesIterator(
                source_length, batch_size, epochs=epochs, iterations=iterations)
        else:
            # both of these are not needed for a SeqeuntialIndicesIterator
            if epochs is not None or iterations is not None:
                raise Exception
            indices_iterator = SequentialIndicesIterator(source_length,
                                                         batch_size)
        return indices_iterator

    def __iter__(self):
        return self

    def __next__(self):
        """
        Gets the next batch and pops the corresponding iterator state for that
        batch
        """
        batch = next(self.dataloaderiter)
        self.multi_batch_indices_iterator =\
            self.queue_wrapped_indices_iterator.pop_iterator()
        return batch

    @property
    def indices_iterator(self):
        """
        Gets the indices_iterator with the correct current state

        Note: this is wrapped by multi_batch_indices_iterator, but the input to
        this object is the unwrapped iterator, a subclass of
        AbstractIndicesIterator unlike MultiBatchIndicesIterator
        """
        return self.multi_batch_indices_iterator.indices_iterator

    def iterator_info(self):
        return self.multi_batch_indices_iterator.iterator_info()

    def take_step(self):
        """
        Checks whether the returned batch is the final subbatch in the actual
        batch
        """
        return self.multi_batch_indices_iterator.take_step()

class CollateFnObject:
    """
    A callable object implementing the collate function by calling the batch
    function of the batcher

    Note: this is needed to create the DataLoader
    """
    def __init__(self, batcher, devices):
        self.batcher = batcher
        self.devices = devices

    def __call__(self, instances):
        return self.batcher.batch(instances, devices=self.devices)

class DatasetWrapper(Dataset):
    """
    Implementation of a Dataset that uses the input batcher to process each
    datapoint

    Note: This is needed to create the DataLoader
    """
    def __init__(self, dataset, batcher):
        self.dataset = dataset
        self.batcher = batcher

    def __getitem__(self, index):
        """
        Returns the processed datapoint (an Instance) at index index using the
        batcher
        """
        return self.batcher.process_datapoint(self.dataset[index])

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.dataset)

class IteratorQueueWrapper:
    """
    Keeps track of a queue of the iterator's state after each next call
    """
    def __init__(self, iterator):
        self.iterators = queue.Queue()
        self.last_iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        """
        Adds a copy of the current iterator to the queue and increments it,
        returning it's next output
        """
        output = next(self.last_iterator)
        self.iterators.put(copy.deepcopy(self.last_iterator))
        return output

    def pop_iterator(self):
        """
        Returns the next iterator in the queue
        """
        return self.iterators.get()

# TODO: add more comments
class MultiBatchIndicesIterator:
    """
    Wraps the indices iterator and only returns indices relevant to the current
    process at each next call.  It also allows for batches to be split
    sequentially with delayed updates inside each process.
    """
    def __init__(self, indices_iterator, subbatches_per_process=None):
        self.indices_iterator = indices_iterator
        self.indices_iterator_lookahead = copy.deepcopy(self.indices_iterator)
        self.subbatches_per_process = subbatches_per_process
        self.subbatches_seen = 0
        self.samples_in_batch_seen = 0
        if dist.is_initialized():
            self.rank, self.worldsize = dist.get_rank(), dist.get_world_size()
        self.current_batch_indices = None
        self.current_process_indices = None

    def __iter__(self):
        return self

    def __next__(self):
        """
        Selects only the subset of the batch as a function of the rank and
        worldsize of the current process group
        """
        if self.take_step():
            self.subbatches_seen = 0
            self.samples_in_batch_seen = 0
        if self.subbatches_seen == 0:
            self.current_batch_indices = next(self.indices_iterator_lookahead)
            if dist.is_initialized():
                i, j = split_range(len(self.current_batch_indices),
                                   self.worldsize, self.rank)
                self.current_process_indices = self.current_batch_indices[i:j]
            else:
                self.current_process_indices = self.current_batch_indices
        if self.subbatches_per_process is not None:
            i, j = split_range(len(self.current_process_indices),
                               self.subbatches_per_process,
                               self.subbatches_seen)
            indices = self.current_process_indices[i:j]
        else:
            indices = self.current_process_indices
        self.subbatches_seen += 1
        self.samples_in_subbatch = len(indices)
        self.samples_in_batch_seen += len(indices)
        if self.take_step():
            next(self.indices_iterator)
        return indices

    def take_step(self):
        return self.subbatches_per_process is None\
               or self.subbatches_seen >= self.subbatches_per_process

    def iterator_info(self):
        info = self.indices_iterator.iterator_info()
        info["take_step"] = self.take_step()
        if self.subbatches_per_process is not None:
            info["samples_in_subbatch"] = self.samples_in_subbatch
            info["subbatches_per_process"] = self.subbatches_per_process
        if dist.is_initialized():
            info["rank"] = self.rank
            info["worldsize"] = self.worldsize
            info["samples_per_process"] = len(self.current_process_indices)
        info["samples_in_batch"] = len(self.current_batch_indices)
        return info


class SequentialIndicesIterator(AbstractIndicesIterator):
    """
    Implementation of an AbstractIndicesIterator that iterates over the dataset
    in order

    All functions without comments are described in the superclass.
    """
    def __init__(self, source_length, batch_size):
        self.samples_seen = 0
        self.batches_seen = 0
        self.source_length = source_length
        self.sample_iter = iter(SequentialSampler(range(source_length)))
        self.set_batch_iter(batch_size)

    def __next__(self):
        indices = next(self.batch_iter)
        self.samples_seen += len(indices)
        self.batches_seen += 1
        return indices

    def __len__(self):
        return self.source_length

    def iterator_info(self):
        return {
            "batches_seen":self.batches_seen, "samples_seen":self.samples_seen}

    def set_batch_iter(self, batch_size):
        """
        Sets the batch iterator to a new batch iterator with batch size
        batch_size
        """
        self.batch_size = batch_size
        self.batch_iter = BatchSampleIterator(
            self.sample_iter, batch_size, False)

class RandomIndicesIterator(AbstractIndicesIterator):
    """
    Implementation of an AbstractIndicesIterator which allows one to specify
    various types of random samplers:
        if epochs is None and isinstance(iterations, int):
            the object is a random sampler with replacement that stops after
                iterations iterations
        elif isinstance(epochs, bool) and epochs == True\
            and isinstance(iterations, int):
            the object is a random sampler without replacement that stops after
                iterations iterations and counts epochs
        elif isinstance(epochs, int) and iterations is None:
            the object is a random sampler without replacement that stops after
                epochs epochs and counts iterations
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
        self.replacement = self.get_replacement(
            epochs, iterations)
        self.source_length = source_length
        self.sampler = RandomSampler(range(source_length),
                                     replacement=self.replacement)
        self.sample_iter = iter(self.sampler)
        self.set_batch_iter(batch_size)

    def __next__(self):
        try:
            indices = next(self.batch_iter)
        except StopIteration:
            self.epochs_seen += 1
            if isinstance(self.num_epochs, int)\
               and self.epochs_seen >= self.num_epochs:
                raise StopIteration
            self.sample_iter = iter(self.sampler)
            self.set_batch_iter(self.batch_size)
            indices = next(self.batch_iter)
        if isinstance(self.num_iterations, int)\
           and self.batches_seen >= self.num_iterations:
            raise StopIteration
        self.samples_seen += len(indices)
        self.batches_seen += 1
        return indices

    def __len__(self):
        if (self.num_epochs is None and isinstance(self.num_iterations, int))\
           or (isinstance(self.num_epochs, bool) and self.num_epochs == True
           and isinstance(self.num_iterations, int)):
            return self.num_iterations
        elif isinstance(self.num_epochs, int) and self.num_iterations is None:
            return math.ceil(
                (self.num_epochs*self.source_length - self.samples_seen)\
                /min(self.batch_size, self.source_length))+self.batches_seen

    def iterator_info(self):
        return {"batches_seen":self.batches_seen,
                "samples_seen":self.samples_seen,
                "epochs_seen":self.epochs_seen}

    def set_batch_iter(self, batch_size):
        """
        Sets the batch iterator to a new batch iterator with batch size
        batch_size
        """
        self.batch_size = batch_size
        self.batch_iter = BatchSampleIterator(
            self.sample_iter, batch_size, False)

    def set_stop(self, epochs=None, iterations=None):
        """
        Changes the stopping point of the iterator by respecifying the epochs
        and iterations parameters

        This is only allowed if the epochs and iterations are still compatable
        with the current sampler, meaning they don't change a with replacement
        to a without replacement sampler or vice versa.
        """
        if self.get_replacement(epochs, iterations) != self.replacement:
            raise Exception
        self.num_epochs = epochs
        self.num_iterations = iterations

    def get_replacement(self, epochs, iterations):
        """
        Checks if the flags are compatable and returns if the random sampling is
        done with replacement
        """
        if epochs is None and isinstance(iterations, int):
            return True
        elif (isinstance(epochs, bool) and epochs == True\
             and isinstance(iterations, int)) or (isinstance(epochs, int)\
             and iterations is None):
            return False
        else:
            raise Exception

class BatchSampleIterator:
    """
    Iterates over batches of indices using the iterator of a sampler (similar to
    what BatchSampler.__iter__() returns, but it is an actual iterator rather
    than a generator)
    """
    def __init__(self, sample_iter, batch_size, drop_last):
        self.sample_iter = sample_iter
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next batch of indices using the sample_iter
        """
        batch = []
        for idx in self.sample_iter:
            batch.append(idx)
            if len(batch) >= self.batch_size:
                break
        if len(batch) == 0 or (self.drop_last and len(batch) < self.batch_size):
            raise StopIteration
        return batch

class StandardInstance(AbstractInstance):
    """
    Implementation of an AbstractInstance that does no processing

    All functions without comments are described in the superclass.
    """
    def __init__(self, raw_datapoint, device=None):
        self.datapoint = raw_datapoint
        self.input = {k:torch.tensor(v).to(device=device)
                      for k,v in raw_datapoint.items()}

class StandardBatch(AbstractBatch):
    """
    Implementation of a simple AbstractBatch

    All functions without comments are described in the superclass.
    """
    def __init__(self, instances):
        self.datapoints = [instance.datapoint for instance in instances]
        self.inputs = {k:pad_and_concat([instance.input[k]
                                         for instance in instances])
                       for k in instances[0].input.keys()}

# TODO: figure out if something can be done for the output batch and output
# instance objects

# helper functions used to create a StandardBatch object

def get_max_dims(tensors):
    """
    Returns None if the tensors are all the same size and the maximum size in
    each dimension otherwise
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