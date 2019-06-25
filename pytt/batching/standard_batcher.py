import torch
from torch.nn import functional as F
from pytt.batching.abstract_batcher import AbstractBatcher,\
                                           AbstractInstance,\
                                           AbstractBatch
from pytt.batching.standard_batch_iterator import StandardBatchIterator


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

    def batch(self, instances, devices=None, batch_class=None):
        if batch_class is None:
            batch_class = StandardBatch
        if devices is None:
            return batch_class(list(instances))
        else:
            if isinstance(devices, str):
                return batch_class(list(instances)).to(devices)
            else:
                return batch_class.init_batches_across_devices(
                    list(instances), devices)

    def out_batch_to_readable(self, output_batch):
        # TODO: implement this
        raise NotImplementedError

    def batch_iterator(self, dataset, indices_iterator, subbatches=None,
                       num_workers=0, devices=None):
        """
        TODO: fill out this comment
        """
        return StandardBatchIterator(
            self,
            dataset,
            indices_iterator,
            subbatches=subbatches,
            num_workers=num_workers,
            devices=devices,
        )


class StandardInstance(AbstractInstance):
    """
    Implementation of an AbstractInstance that does no processing

    All functions without comments are described in the superclass.
    """
    def __init__(self, raw_datapoint, device=None):
        self.datapoint = raw_datapoint
        self.tensors = {k:torch.tensor(v).to(device=device)
                        for k,v in raw_datapoint.items()}


class StandardBatch(AbstractBatch):
    """
    Implementation of a simple AbstractBatch

    All functions without comments are described in the superclass.
    """
    def __init__(self, instances):
        self.datapoints = [instance.datapoint for instance in instances]
        self.tensors = {k:pad_and_concat([instance.tensors[k]
                                          for instance in instances])
                        for k in instances[0].tensors.keys()}

    def get_unsupervised(self):
        # NOTE: in most cases this should be overridden!
        return self.tensors

    def get_labels(self):
        # NOTE: in most cases this should be overridden!
        return self.tensors


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
