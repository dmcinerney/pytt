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
from summarization.summarization_dataset import SummarizationDataset, load_vocab
from summarization.summarization_batcher import SummarizationBatcher
from pytt.nlp.tokenizer import Tokenizer
from torch import nn
from pytt.utils import get_random_state, seed_state
from pytt.distributed import distributed_wrapper, setup
from torch.nn.parallel import DistributedDataParallel as DDP
from fairseq.legacy_distributed_data_parallel import LegacyDistributedDataParallel as LDDP
from torch.optim import Adam
from pytt.training.trainer import Trainer
from pytt.logger import logger

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.p = nn.Linear(1,1)
    
    def forward(self, batch):
        batch.to(next(iter(self.p.parameters())).device)
#         print("forward")
        # print(self.p.weight)
        return dict(output=self.p(batch.inputs['text'][:, :1].float()))

def loss_func(output):
    return output.sum()

def iterate(batcher, raw_dataset):
    rank_worldsize = (torch.distributed.get_rank(), torch.distributed.get_world_size()) if torch.distributed.is_initialized() else None
    print(rank_worldsize)
    model = Model().cuda()
    if rank_worldsize is not None:
        # model = DDP(model.to(rank_worldsize[0]), device_ids=[rank_worldsize[0]])
        model = LDDP(model.to(rank_worldsize[0]), rank_worldsize[1])
    optimizer = Adam([p for p in model.parameters()])
    batch_iterator = batcher.batch_iterator(raw_dataset, batch_size=15, subbatches=2, random=True, iterations=200, num_workers=5)#, devices=['cuda:0', 'cuda:1'])
    for batch in batch_iterator:
#         print(batch)
        print(batch_iterator.iterator_info(), len(batch_iterator.indices_iterator), len(raw_dataset))
        if rank_worldsize is not None:
            batch.to('cuda:'+str(rank_worldsize[0]))
        else:
            batch.to('cuda')
        optimizer.zero_grad()
        model(batch).backward()
        optimizer.step()
        if (batch_iterator.iterator_info()['batches_seen']+1) % 1000 == 0:
            break
    print("done")
    batch_iterator.indices_iterator.set_stop(iterations=400)
    batch_iterator = batcher.batch_iterator(raw_dataset, indices_iterator=batch_iterator.indices_iterator, num_workers=5)#, devices=['cuda:0', 'cuda:1'])
    for batch in batch_iterator:
        print(batch_iterator.iterator_info(), len(batch_iterator.indices_iterator), len(raw_dataset))
        # print(batch)
        if rank_worldsize is not None:
            batch.to('cuda:'+str(rank_worldsize[0]))
        else:
            batch.to('cuda')
        optimizer.zero_grad()
        model(batch).backward()
        optimizer.step()
        if (batch_iterator.iterator_info()['batches_seen']+1) % 1000 == 0:
            break
    print("done")

# if __name__ == '__main__':
#     seed_state()
# #     torch.multiprocessing.set_start_method("spawn")
#     raw_dataset = SummarizationDataset('/home/jered/Documents/Projects/Summarization/data/cnn_dataset/val_processed.data')
#     tokenizer = Tokenizer(load_vocab('/home/jered/Documents/Projects/Summarization/data/cnn_dataset/vocab', 50000))
#     batcher = SummarizationBatcher(tokenizer)
#     nprocs = 2
#     distributed_iterate = distributed_wrapper(iterate, nprocs, random_state=get_random_state())
#     distributed_iterate(batcher, raw_dataset)
# #     setup(0,1)
#     # iterate(batcher, raw_dataset)
def spawn_function():
    model = Model()
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        worldsize = torch.distributed.get_world_size()
        model = LDDP(model.to(rank), worldsize)
    optimizer = Adam([p for p in model.parameters()])
    raw_dataset = SummarizationDataset('/home/jered/Documents/Projects/Summarization/data/cnn_dataset/val_processed.data')
    tokenizer = Tokenizer(load_vocab('/home/jered/Documents/Projects/Summarization/data/cnn_dataset/vocab', 50000))
    batcher = SummarizationBatcher(tokenizer)
    batch_iterator = batcher.batch_iterator(raw_dataset, batch_size=15, subbatches=None, random=True, iterations=200, num_workers=5)
    trainer = Trainer(model, optimizer, batch_iterator)
    logger.set_verbosity(2)
    trainer.train(loss_func)

if __name__ == '__main__':
    seed_state()
#     torch.multiprocessing.set_start_method("spawn")
    nprocs = 2
    distributed_spawn_function = distributed_wrapper(spawn_function, nprocs, random_state=get_random_state())
    distributed_spawn_function()
    # setup(0,1)
    # spawn_function()
