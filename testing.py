from preprocessing.raw_dataset import RawDataset
from batching.standard_batcher import StandardBatcher
from batching.abstract_batcher import AbstractIndicesIterator
raw_dataset = RawDataset('test_dataset.csv')
raw_dataset.df = raw_dataset.df[:10]
batcher = StandardBatcher()
batch = batcher.batch_from_dataset(raw_dataset, [0,1])
batch_iterator = batcher.batch_iterator(raw_dataset, batch_size=2, random=True, epochs=2, num_workers=4)
for batch in batch_iterator:
    print(batch_iterator.iterator_info(), batch_iterator.indices_iterator.replacement, len(batch_iterator.indices_iterator), len(raw_dataset))
    if (batch_iterator.iterator_info()['batches_seen']+1) % 7 == 0:
        break
print("saving the indices iterator")
batch_iterator.indices_iterator.save('indices_iterator.pkl')
print("loading the indices iterator")
indices_iterator = AbstractIndicesIterator.load('indices_iterator.pkl')
batch_iterator2 = batcher.batch_iterator(raw_dataset, indices_iterator=indices_iterator)
for batch in batch_iterator2:
    print(batch_iterator2.iterator_info(), batch_iterator2.indices_iterator.replacement, len(batch_iterator2.indices_iterator), len(raw_dataset))
