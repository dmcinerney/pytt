from pytt.batching.abstract_batcher import AbstractInstance
from pytt.batching.standard_batcher import StandardBatcher, StandardBatch
import torch

class SummarizationBatcher(StandardBatcher):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def out_batch_to_readable(self, output_batch):
        # TODO: implement this
        raise NotImplementedError

class TrainSummarizationBatcher(SummarizationBatcher):
    def process_datapoint(self, raw_datapoint):
        return TrainSummarizationInstance(raw_datapoint, self.tokenizer)

class TestSummarizationBatcher(SummarizationBatcher):
    def process_datapoint(self, raw_datapoint):
        return TestSummarizationInstance(raw_datapoint, self.tokenizer)

class AbstractSummarizationInstance(AbstractInstance):
    def __init__(self, raw_datapoint, tokenizer):
        text, oov_token2id = tokenizer.tokens2tensor(["<start>"]+raw_datapoint['text']+["<end>"])
        summary, _ = tokenizer.tokens2tensor(["<start>"]+raw_datapoint['summary']+["<end>"],
                                             oov_token2id=oov_token2id)
        self.datapoint = {"raw_datapoint":raw_datapoint,
                          "oov_token2id":oov_token2id}
        self.tensors = {
            'text':text,
            'text_length':torch.tensor(len(text)),
            'summary': summary,
            'summary_length':torch.tensor(len(summary)),
        }

class TrainSummarizationInstance(AbstractSummarizationInstance):
    def __init__(self, raw_datapoint, tokenizer):
        super(TrainSummarizationInstance, self).__init__(raw_datapoint, tokenizer)
        self.observed_keys = ['text', 'text_length', 'summary',
                              'summary_length']
        self.target_keys = []

class TestSummarizationInstance(AbstractSummarizationInstance):
    def __init__(self, raw_datapoint, tokenizer):
        super(TestSummarizationInstance, self).__init__(raw_datapoint, tokenizer)
        self.observed_keys = ['text', 'text_length']
        self.target_keys = ['summary', 'summary_length']
