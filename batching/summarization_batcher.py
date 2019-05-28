from abstract_batcher import AbstractInstance
from standard_batcher import StandardBatcher, StandardBatch

class SummarizationBatcher(StandardBatcher):
	def __init__(self, tokenizer):
        self.tokenizer

    def process_datapoint(self, raw_datapoint):
        return SummarizationInstance(raw_datapoint, tokenizer)

    def out_batch_to_readable(self, output_batch):
        # TODO: implement this
        raise NotImplementedError

class SummarizationInstance(AbstractInstance):
	def __init__(raw_datapoint, tokenizer):
		self.raw_datapoint = raw_datapoint
		self.processed_datapoint = {k:tokenizer.tokenize(v) for k,v in raw_datapoint.items()}