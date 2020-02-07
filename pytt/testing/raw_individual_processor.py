from pytt.testing.datapoint_processor import DatapointProcessor

class RawIndividualProcessor(DatapointProcessor):
    def __init__(self, model, postprocessor, batcher, device=None):
        super(RawIndividualProcessor, self).__init__(model, postprocessor)
        self.batcher = batcher
        self.device = device

    def process_datapoint(self, raw_datapoint):
        batch = self.batcher.batch_from_raw([raw_datapoint],
                                            devices=self.device)
        return super(RawIndividualProcessor, self).process_batch(batch)
