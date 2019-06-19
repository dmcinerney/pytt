from pytt.batching.abstract_batcher import AbstractInstance
from pytt.batching.standard_batcher import StandardBatcher, StandardBatch
import torch

class SummarizationBatcher(StandardBatcher):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process_datapoint(self, raw_datapoint):
        return SummarizationInstance(raw_datapoint, self.tokenizer)

    def out_batch_to_readable(self, output_batch):
        # TODO: implement this
        raise NotImplementedError

class SummarizationInstance(AbstractInstance):
    def __init__(self, raw_datapoint, tokenizer):
        text, oov_token2id = tokenizer.tokens2tensor(raw_datapoint['text'])
        summary, _ = tokenizer.tokens2tensor(raw_datapoint['summary'],
                                             oov_token2id=oov_token2id)
        self.datapoint = {"raw_datapoint":raw_datapoint,
                          "oov_token2id":oov_token2id}
        self.input = {
            'text':text,
            'text_length':torch.tensor(len(text)),
            'summary': summary,
            'summary_length':torch.tensor(len(summary)),
        }
