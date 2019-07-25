import copy
import queue
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pytt.batching.abstract_batcher import AbstractBatchIterator
from pytt.batching.indices_iterator import StandardIteratorInfo
from pytt.utils import split_range

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
    def __init__(self, batcher, dataset, indices_iterator, subbatches=None,
                 num_workers=0, devices=None):
        dataset = DatasetWrapper(dataset, batcher)
        # Does nothing when not distributed or split into subbatches
        self.subbatch_indices_iterator = SubbatchIndicesIterator(
            indices_iterator, subbatches_per_process=subbatches)
        # Wraps the iterator in a queue to ensure the correct iterator state is
        #   kept for logging and checkpointing information
        # WARNING: this means that if subbatch_indices_iterator is changed,
        #   the dataloader will not change behavior, one would need to
        #   reinitialize the BatchIterator
        self.queue_wrapped_indices_iterator = IteratorQueueWrapper(
            copy.deepcopy(self.subbatch_indices_iterator))
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

    def __iter__(self):
        return self

    def __next__(self):
        """
        Gets the next batch and pops the corresponding iterator state for that
        batch
        """
        batch = next(self.dataloaderiter)
        self.subbatch_indices_iterator =\
            self.queue_wrapped_indices_iterator.pop_iterator()
        return batch

    @property
    def indices_iterator(self):
        """
        Gets the indices_iterator with the correct current state

        Note: this is wrapped by subbatch_indices_iterator, but the input to
        this object is the unwrapped iterator, a subclass of
        AbstractIndicesIterator unlike SubbatchIndicesIterator
        """
        return self.subbatch_indices_iterator.indices_iterator

    def iterator_info(self):
        return self.subbatch_indices_iterator.iterator_info()

    def take_step(self):
        """
        Checks whether the returned batch is the final subbatch in the actual
        batch
        """
        return self.subbatch_indices_iterator.take_step()

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
class SubbatchIndicesIterator:
    """
    Wraps the indices iterator and only returns indices relevant to the current
    process at each next call.  It also allows for batches to be split
    sequentially with delayed updates inside each process.

    NOTE: an error will be thrown if there are more subbatches than instances
      IMPORTANT: this can happen on the last batch because it is generally
        smaller
      TODO: fix this!
    """
    def __init__(self, indices_iterator, subbatches_per_process=None):
        self.indices_iterator = indices_iterator
        self.indices_iterator_lookahead = copy.deepcopy(self.indices_iterator)
        self.subbatches_per_process = subbatches_per_process
        self.subbatches = None
        self.current_batch_indices = None

    def __iter__(self):
        return self

    def __next__(self):
        """
        Selects a subset of the current batch as defined by the subbatches
        object
        """
        if self.subbatches is None or self.take_step():
            self.current_batch_indices = next(self.indices_iterator_lookahead)
            self.subbatches = Subbatches(
                len(self.current_batch_indices),
                sequential_subbatches=self.subbatches_per_process
            )
        i, j = next(self.subbatches)
        indices = self.current_batch_indices[i:j]
        if self.take_step():
            next(self.indices_iterator)
        return indices

    def take_step(self):
        return self.subbatches.last_subbatch()

    def iterator_info(self):
        info = self.indices_iterator.iterator_info()
        return SubbatchIteratorInfo(
            batches_seen=info.batches_seen,
            total_batches=info.total_batches,
            samples_seen=info.samples_seen,
            subbatches=copy.deepcopy(self.subbatches),
            epochs_seen=info.epochs_seen
        )


# TODO: add comments
class SubbatchIteratorInfo(StandardIteratorInfo):
    def __init__(self, batches_seen, total_batches, samples_seen, subbatches,
                 epochs_seen=None):
        super(SubbatchIteratorInfo, self).__init__(batches_seen, total_batches,
                                                   samples_seen,
                                                   epochs_seen=epochs_seen)
        self.subbatches = subbatches

    def __str__(self):
        base = super(SubbatchIteratorInfo, self).__str__()
        base += ", batch_size: "+str(self.subbatches.samples_in_fullbatch)
        return base

    def subbatch_str(self):
        return str(self.subbatches)

    def to_tensor(self):
        numbers = []
        for name in [
            'batches_seen',
            'total_batches',
            'samples_seen',
            'epochs_seen',
        ]:
            attr = getattr(self, name)
            if attr is not None:
                numbers.append(attr)
        return torch.cat((torch.tensor(numbers),
                          self.subbatches.to_tensor()), 0)

    def from_tensor(self, tensor, isiter=False):
        if not isiter:
            tensor_iter = iter(tensor)
        else:
            tensor_iter = tensor
        for name in [
            'batches_seen',
            'total_batches',
            'samples_seen',
            'epochs_seen',
        ]:
            if getattr(self, name) is not None:
                setattr(self, name, int(next(tensor_iter).item()))
        self.subbatches.from_tensor(tensor_iter, isiter=True)
        return self

class Subbatches:
    def __init__(self, samples_in_fullbatch, sequential_subbatches=None):
        self.samples_in_subbatch = None
        self.samples_in_fullbatch = samples_in_fullbatch
        self.sequential_subbatch_num = -1
        self.max_sequential_subbatches = sequential_subbatches\
            if sequential_subbatches is not None else 1
        if dist.is_initialized():
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        else:
            self.rank, self.world_size = None, None
        i, j = self.peek(rank=self.rank)
        self.sequential_subbatches = min(
            self.max_sequential_subbatches, j-i)

    def peek(self, rank=None, sequential_subbatch_num=None):
        i, j = 0, self.samples_in_fullbatch
        if rank is not None:
            if not dist.is_initialized():
                raise Exception
            i, j = split_range(j-i, self.world_size, self.rank)
        if sequential_subbatch_num is not None:
            offset = i
            i, j = split_range(j-i, self.max_sequential_subbatches,
                               self.sequential_subbatch_num)
            i, j = i+offset, j+offset
        return i, j

    def get_ranks(self):
        if not dist.is_initialized():
            return None
        ranks = []
        for rank in range(self.world_size):
            i, j = self.peek(rank=rank)
            if j-i > 0:
                ranks.append(rank)
        return ranks

    def __next__(self):
        self.sequential_subbatch_num += 1
        if self.sequential_subbatch_num >= self.sequential_subbatches:
            raise StopIteration
        i, j = self.peek(rank=self.rank,
                         sequential_subbatch_num=self.sequential_subbatch_num)
        self.samples_in_subbatch = j-i
        return i, j

    def last_subbatch(self):
        return self.sequential_subbatch_num + 1 == self.sequential_subbatches

    def __str__(self):
        base = "samples_in_subbatch: "+str(self.samples_in_subbatch)
        if self.sequential_subbatch_num is not None:
            base += ", sequential_subbatch_num: "\
                    +str(self.sequential_subbatch_num)\
                    +" of "\
                    +str(self.sequential_subbatches)
        if self.rank is not None:
            base += ", rank: "+str(self.rank)
        return base

    def to_tensor(self):
        numbers = []
        for name in [
            'samples_in_subbatch',
            'samples_in_fullbatch',
            'sequential_subbatch_num',
            'max_sequential_subbatches',
            'rank',
            'world_size',
            'sequential_subbatches'
        ]:
            attr = getattr(self, name)
            if attr is not None:
                numbers.append(attr)
        return torch.tensor(numbers)

    def from_tensor(self, tensor, isiter=False):
        if not isiter:
            tensor_iter = iter(tensor)
        else:
            tensor_iter = tensor
        for name in [
            'samples_in_subbatch',
            'samples_in_fullbatch',
            'sequential_subbatch_num',
            'max_sequential_subbatches',
            'rank',
            'world_size',
            'sequential_subbatches'
        ]:
            if getattr(self, name) is not None:
                setattr(self, name, int(next(tensor_iter).item()))
        return self
