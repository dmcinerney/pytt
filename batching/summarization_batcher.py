from batching.abstract_batcher import AbstractInstance
from batching.standard_batcher import StandardBatcher, StandardBatch

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
        text, oov_indices = tokenizer.tokens2tensor(raw_datapoint['text'])
        summary, _ = tokenizer.tokens2tensor(raw_datapoint['summary'], oov_indices=oov_indices)
        self.datapoint = {"raw_datapoint":raw_datapoint, "oov_indices":oov_indices}
        self.input = {
            'text':text,
            'text_length':torch.tensor(len(text)),
            'summary': summary,
            'summary_length':torch.tensor(len(summary)),
        }
