from pytt.testing.datapoint_processor import DatapointProcessor

class RawIndividualProcessor(DatapointProcessor):
    def __init__(self, model, batcher, batch_info_class, device=None):
        super(RawIndividualProcessor, self).__init__(model, batch_info_class)
        self.batcher = batcher
        self.device = device

    def process_datapoint(self, raw_datapoint):
        batch = self.batcher.batch_from_raw([raw_datapoint])
        if self.device is not None:
            batch = batch.to(self.device)
        return super(RawIndividaulProcessor, self).process_batch(batch)
