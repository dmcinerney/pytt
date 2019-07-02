import torch
from pytt.batching.abstract_batcher import AbstractBatcher,\
                                           AbstractInstance,\
                                           AbstractBatch
from pytt.batching.standard_batch_iterator import StandardBatchIterator
from pytt.utils import pad_and_concat


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
        self.observed_keys = self.tensors.keys()
        self.target_keys = self.tensors.keys()


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
        self.observed_keys = instances[0].observed_keys
        self.target_keys = instances[0].target_keys

    def get_observed(self):
        return {k:self.tensors[k] for k in self.observed_keys}

    def get_target(self):
        return {k:self.tensors[k] for k in self.target_keys}


# TODO: figure out if something can be done for the output batch and output
# instance objects
