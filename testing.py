# from preprocessing.raw_dataset import RawDataset
# from batching.standard_batcher import StandardBatcher
# from batching.abstract_batcher import AbstractIndicesIterator
# raw_dataset = RawDataset('test_dataset.csv')
# raw_dataset.df = raw_dataset.df[:10]
# batcher = StandardBatcher()
# batch = batcher.batch_from_dataset(raw_dataset, [0,1])
# batch_iterator = batcher.batch_iterator(raw_dataset, batch_size=2, random=True, epochs=2, num_workers=4)
# for batch in batch_iterator:
#     print(batch_iterator.iterator_info(), batch_iterator.indices_iterator.replacement, len(batch_iterator.indices_iterator), len(raw_dataset))
#     if (batch_iterator.iterator_info()['batches_seen']+1) % 7 == 0:
#         break
# print("saving the indices iterator")
# batch_iterator.indices_iterator.save('indices_iterator.pkl')
# print("loading the indices iterator")
# indices_iterator = AbstractIndicesIterator.load('indices_iterator.pkl')
# batch_iterator2 = batcher.batch_iterator(raw_dataset, indices_iterator=indices_iterator)
# for batch in batch_iterator2:
#     print(batch_iterator2.iterator_info(), batch_iterator2.indices_iterator.replacement, len(batch_iterator2.indices_iterator), len(raw_dataset))
import torch
from nlp.summarization_dataset import SummarizationDataset, load_vocab
from nlp.summarization_batcher import SummarizationBatcher
from nlp.tokenizer import Tokenizer
from torch import nn

class TestModel(nn.Module):
	def forward(self, **kwargs):
		import pdb; pdb.set_trace()

if __name__ == '__main__':
	# torch.multiprocessing.set_start_method("spawn")
	raw_dataset = SummarizationDataset('/home/jered/Documents/Projects/Summarization/data/cnn_dataset/val_processed.data')
	tokenizer = Tokenizer(load_vocab('/home/jered/Documents/Projects/Summarization/data/cnn_dataset/vocab', 50000))
	batcher = SummarizationBatcher(tokenizer)
	batch_iterator = batcher.batch_iterator(raw_dataset, batch_size=16, random=True, iterations=200, num_workers=5)#, devices=['cuda:0', 'cuda:1'])
	for batch in batch_iterator:
	    print(batch_iterator.iterator_info(), batch_iterator.indices_iterator.replacement, len(batch_iterator.indices_iterator), len(raw_dataset))
	    # print(batch)
	    if (batch_iterator.iterator_info()['batches_seen']+1) % 1000 == 0:
	        break
	print("done")
	batch_iterator.indices_iterator.set_stop(iterations=400)
	batch_iterator = batcher.batch_iterator(raw_dataset, indices_iterator=batch_iterator.indices_iterator, num_workers=5)#, devices=['cuda:0', 'cuda:1'])
	for batch in batch_iterator:
	    print(batch_iterator.iterator_info(), batch_iterator.indices_iterator.replacement, len(batch_iterator.indices_iterator), len(raw_dataset))
	    # print(batch)
	    if (batch_iterator.iterator_info()['batches_seen']+1) % 1000 == 0:
	        break
	print("done")