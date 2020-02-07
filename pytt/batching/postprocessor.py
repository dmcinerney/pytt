import torch
from pytt.utils import IndexIter, pad_and_concat

class AbstractPostprocessor:
    def __init__(self):
        """
        TODO: fill out this comment
        """
        raise NotImplementedError

    def output_batch(self, batch, outputs):
        """
        Returns a batch info object that performs any postprocessing of the
        model outputs
        """
        raise NotImplementedError

# TODO: determine if the following are necessary


class AbstractOutputBatch:
    """
    Handles collecting model outputs into an output_batch object

    This is an abstract class that allows a very flexible framework for creating
    output batches.  See standard_batchers.py for examples of standard
    implementations of this abstract architecture.
    """
    @classmethod
    def from_outputs(cls, batch, outputs):
        """
        TODO: fill out this comment
        """
        raise NotImplementedError

    @classmethod
    def loss(cls, batch, outputs):
        """
        TODO: fill out this comment
        """
        raise NotImplementedError

    @classmethod
    def stats(cls, batch, outputs):
        """
        TODO: fill out this comment
        """
        raise NotImplementedError

    def __init__(self, batch_length, batch_stats, batch=None, outputs=None):
        self.batch_length = batch_length
        self.batch_stats = batch_stats
        self.batch = batch
        self.outputs = outputs

    def __add__(self, output_batch):
        """
        TODO: fill out this comment
        """
        raise NotImplementedError

    def __radd__(self, i):
        if i != 0:
            raise Exception
        return self

    def __str__(self):
        """
        TODO: fill out this comment
        """
        raise NotImplementedError

    def get_output_insances(self):
        """
        Returns a list of AbstractOutputInstance objects
        """
        raise NotImplementedError

# TODO: currently unused
class AbstractOutputInstance:
    """
    Handles postprocessing of model ouptuts into something readable

    This is an abstract class that allows a very flexible framework for creating
    output instances.  See standard_batchers.py for examples of standard
    implementations of this abstract architecture.
    """
    def __init__(self, output):
        """
        Initializes the output instance with the output for a single example
        """
        raise NotImplementedError

class StandardPostprocessor(AbstractPostprocessor):
    def __init__(self):
        pass

    def output_batch(self, batch, outputs, output_batch_class=None):
        if output_batch_class is None:
            output_batch_class = StandardOutputBatch
        return output_batch_class.from_outputs(batch, outputs)

class StandardOutputBatch(AbstractOutputBatch):
    @classmethod
    def from_outputs(cls, batch, outputs):
        loss, stats = cls.get_loss_stats(batch, outputs)
        # this might be called when gred is enabled, and there is no reason
        # to enable grad when not necessary
        with torch.autograd.no_grad():
            return loss, cls(len(batch), stats, batch=batch, outputs=outputs)

    @classmethod
    def get_loss_stats(cls, *args, **kwargs):
        loss = cls.loss(*args, **kwargs)
        # this might be called when gred is enabled, and there is no reason
        # to enable grad when not necessary
        with torch.autograd.no_grad():
            stats = cls.stats(*args, **kwargs)
        if loss is not None:
            stats['loss'] = loss.item()
        return loss, stats

    @classmethod
    def loss(cls, batch, outputs):
        return None

    @classmethod
    def stats(cls, batch, outputs):
        return {}

    def __init__(self, batch_length, batch_stats, batch=None, outputs=None):
        self.batch_length = batch_length
        self.batch_stats = batch_stats
        # defaults to not saving batch our outputs even if passed in
        # (can be overridden)
        self.batch = None
        self.outputs = None

    def write_to_tensorboard(self, writer, iterator_info):
        # NOTE: this only occurs after every iteration, which in some cases is
        #       after two output_batch's have been added together
        global_step = iterator_info.batches_seen
        for k,v in self.batch_stats.items():
            writer.add_scalar(k, v/self.batch_length, global_step)

    def __add__(self, output_batch):
        new_batch_stats = {}
        for k in set(self.batch_stats.keys()).union(
                     output_batch.batch_stats.keys()):
            new_batch_stats[k] = self.batch_stats[k]\
                                     + output_batch.batch_stats[k]
        new_batch_length = self.batch_length + output_batch.batch_length
        if self.outputs is not None:
            new_outputs = {}
            for k in set(self.outputs.keys()).union(
                         output_batch.outputs.keys()):
                new_outputs[k] = pad_and_concat(
                    [self.outputs[k],
                     output_batch.outputs[k]])
                new_outputs[k] = new_outputs[k].view(
                    -1, *new_outputs[k].shape[2:])
        else:
            new_outputs = None
        if self.batch is not None:
            raise NotImplementedError
        else:
            new_batch = None
        return self.__class__(new_batch_length,
                              new_batch_stats,
                              batch=new_batch,
                              outputs=new_outputs)

    def __radd__(self, i):
        if i != 0:
            raise Exception
        return self

    def __str__(self):
        step_info = ""
        first = True
        for (k,v) in sorted(self.batch_stats.items(), key=lambda kv: kv[0]):
            if not first:
                step_info += ", "
            first = False
            step_info += ("%s per instance: " % k)\
                         +str(v/self.batch_length)
        return step_info

    def to_tensor(self):
        # NOTE: needed to transport batch info across multiple threads
        if self.outputs is not None or self.batch is not None:
            raise NotImplementedError
        list_of_floats = []
        list_of_floats.append(self.batch_length)
        for k,v in sorted(self.batch_stats.items(),
                          key=lambda kv: kv[0]):
            list_of_floats.append(v)
        return torch.tensor(list_of_floats)

    def from_tensor(self, tensor, index_iter=None):
        # NOTE: needed to transport batch info across multiple threads
        if index_iter is None:
            index_iter = IndexIter(0,tensor.size(0))
        self.batch_length = int(tensor[next(index_iter)].item())
        for k,v in sorted(self.batch_stats.items(),
                          key=lambda kv: kv[0]):
            self.batch_stats[k] = float(tensor[next(index_iter)].item())
        return self
