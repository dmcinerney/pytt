from pytt.testing.datapoint_processor import DatapointProcessor

class RawIndividualProcessor(DatapointProcessor):
    def __init__(self, model, batcher, test_func, device=None):
        self.model = model
        self.batcher = batcher
        self.test_func = test_func
        self.device = device

    def process_datapoint(self, raw_datapoint):
        batch = self.batcher.batch_from_raw([raw_datapoint])
        if self.device is not None:
            batch = batch.to(self.device)
        return super(RawIndividaulProcessor, self).process_batch(batch, self.test_func)
