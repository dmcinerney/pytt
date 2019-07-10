import math
from torch.utils.data import SequentialSampler, RandomSampler
from pytt.batching.abstract_batcher import AbstractIndicesIterator, IteratorInfo


def init_indices_iterator(source_length, batch_size, random=None,
                          epochs=None, iterations=None):
    """
    Initializes some common types of indices_iterators (only called if one
    is not given)
    """
    # Checks that the batch_size is set because it is needed for both
    #   IndicesIterator types
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
        return math.ceil(self.source_length/self.batch_size)

    def iterator_info(self):
        return StandardIteratorInfo(self.batches_seen, len(self),
                                    self.samples_seen)

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
        self.epochs_seen = None if epochs is None or epochs == False else 0
        self.num_epochs = epochs
        self.num_iterations = iterations
        self.replacement = self.get_replacement(epochs, iterations)
        self.source_length = source_length
        self.sampler = RandomSampler(range(source_length),
                                     replacement=self.replacement)
        self.sample_iter = iter(self.sampler)
        self.set_batch_iter(batch_size)

    def __next__(self):
        try:
            indices = next(self.batch_iter)
        except StopIteration:
            if isinstance(self.num_epochs, int)\
               and not isinstance(self.num_epochs, bool)\
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
        if self.epochs_seen is not None\
           and (self.samples_seen % self.source_length) == 0:
            self.epochs_seen += 1
        return indices

    def __len__(self):
        if (self.num_epochs is None and isinstance(self.num_iterations, int))\
           or (isinstance(self.num_epochs, bool) and self.num_epochs == True
           and isinstance(self.num_iterations, int)):
            return self.num_iterations
        elif isinstance(self.num_epochs, int) and self.num_iterations is None:
            batches_to_go_in_current_epoch = math.ceil(
                (self.source_length - (self.samples_seen % self.source_length))
                /self.batch_size
            )
            batches_to_go_in_future_epochs = \
                (self.num_epochs-self.epochs_seen-1)\
                *math.ceil(self.source_length/self.batch_size)
            return self.batches_seen\
                   +batches_to_go_in_current_epoch\
                   +batches_to_go_in_future_epochs

    def iterator_info(self):
        return StandardIteratorInfo(self.batches_seen, len(self),
                                    self.samples_seen,
                                    epochs_seen=self.epochs_seen)

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
              and isinstance(iterations, int))\
             or (isinstance(epochs, int) and iterations is None):
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


class StandardIteratorInfo(IteratorInfo):
    def __init__(self, batches_seen, total_batches, samples_seen,
                 epochs_seen=None):
        super(StandardIteratorInfo, self).__init__(batches_seen, samples_seen)
        self.total_batches = total_batches
        self.epochs_seen = epochs_seen

    def __str__(self):
        base = "batches_seen: "+str(self.batches_seen)\
             + " of "+str(self.total_batches)\
             + ", samples_seen: "+str(self.samples_seen)
        if self.epochs_seen is not None:
            base = "epochs_seen: "+str(self.epochs_seen)+", "+base
        return base
