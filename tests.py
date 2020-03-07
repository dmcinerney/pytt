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
from summarization.summarization_batcher import TrainSummarizationBatcher, TestSummarizationBatcher
from pytt.batching.indices_iterator import init_indices_iterator
from pytt.nlp.tokenizer import Tokenizer
from torch import nn
from pytt.utils import get_random_state, seed_state
from pytt.distributed import distributed_wrapper, setup, log_bool
from torch.nn.parallel import DistributedDataParallel as DDP
from fairseq.legacy_distributed_data_parallel import LegacyDistributedDataParallel as LDDP
from torch.optim import Adam
from pytt.training.trainer import Trainer
from pytt.training.tracker import Tracker, create_tensorboard_attachment_generator
from pytt.logger import logger
from pytt.training.training_controller import AbstractTrainingController
from pytt.testing.tester import Tester
from pytt.batching.postprocessor import StandardPostprocessor, StandardOutputBatch
#from pytt.setup import Setup
from pytt.email import EmailSender, check_attachment_error

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.p = nn.Linear(1,1)

    def forward(self, text, **kwargs):
        text = text.to(next(iter(self.p.parameters())).device)
#         print("forward")
        # print(self.p.weight)
        output = self.p(text[:, :1].float())
        output = output + torch.randn_like(output)/3
        return dict(output=output)

def loss_func(output):
    return output.sum()

def error_func(*args, **kwargs):
    return {'error': torch.tensor(2*next(iter(kwargs.values())).shape[0])}

class Postprocessor(StandardPostprocessor):
    def output_batch(self, batch, output):
        return super(Postprocessor, self).output_batch(batch, output, output_batch_class=OutputBatch)

class OutputBatch(StandardOutputBatch):
    @classmethod
    def stats(cls, batch, outputs):
        return {k:v.item() for k,v in error_func(**outputs).items()}

    @classmethod
    def loss(cls, batch, outputs):
        return loss_func(**outputs)

def spawn_function(email_sender):
    seed_state()
    model = Model()
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        worldsize = torch.distributed.get_world_size()
#        model = LDDP(model.to('cpu'), worldsize)
        model = LDDP(model.to(rank), worldsize)
    tokenizer = Tokenizer(load_vocab('/home/jered/Documents/data/cnn_dataset/vocab', 50000))
    postprocessor = Postprocessor()
    batcher = TrainSummarizationBatcher(tokenizer)
    val_batcher = TestSummarizationBatcher(tokenizer)
    val_dataset = SummarizationDataset('/home/jered/Documents/data/cnn_dataset/preprocessed/val_processed.data')
    train_dataset = SummarizationDataset('/home/jered/Documents/data/cnn_dataset/preprocessed/val_processed.data')
#    batch_iterator = batcher.batch_iterator(train_dataset, init_indices_iterator(len(train_dataset), batch_size=15, random=True, epochs=2), subbatches=None)
    batch_iterator = batcher.batch_iterator(train_dataset, init_indices_iterator(len(train_dataset), batch_size=15, random=True, iterations=200), subbatches=None)
    val_iterator = batcher.batch_iterator(val_dataset, init_indices_iterator(100, batch_size=15, random=True, iterations=len(batch_iterator.indices_iterator)), subbatches=None)
    optimizer = Adam([p for p in model.parameters()])
    tracker = Tracker(print_every=10, checkpoint_folder='test', checkpoint_every=7, copy_checkpoint_every=7) #, email_every=10, email_sender=email_sender)
    trainer = Trainer(model, postprocessor, optimizer, batch_iterator, val_iterator=val_iterator, tracker=tracker)
    logger.set_verbosity(2)
    trainer.train() #, use_pbar=False)
    if log_bool():
        logger.log("\n\nTESTING")
    val_iterator = batcher.batch_iterator(val_dataset, init_indices_iterator(100, batch_size=15), subbatches=None)
    tester = Tester(model, postprocessor, val_iterator)
    tester.test()
# def spawn_function():
#     setup = Setup(Model)
#     val_dataset = SummarizationDataset('/home/jered/Documents/Projects/Summarization/data/cnn_daily_mail_dataset/val_processed.data')
#     train_dataset = SummarizationDataset('/home/jered/Documents/Projects/Summarization/data/cnn_daily_mail_dataset/val_processed.data')
#     tokenizer = Tokenizer(load_vocab('/home/jered/Documents/Projects/Summarization/data/cnn_daily_mail_dataset/vocab', 50000))
#     batcher = TrainSummarizationBatcher(tokenizer)
#     training_parameters = {
#         'model_parameters':{},
#         'optimizer_parameters':{},
#         'train_iterator_parameters':{
#             'indices_iterator_parameters':{'batch_size':15, 'random':True, 'iterations':200},
#             'batch_iterator_parameters':{},
#         },
#         'val_iterator_parameters':{
#             'indices_iterator_parameters':{'batch_size':15, 'random':True},
#             'batch_iterator_parameters':{},
#         },
#         'trainer_parameters':{'print_every':10, 'checkpoint_folder':'test'},
#     }
#     logger.set_verbosity(1)
#     train_state = setup.train(training_parameters,
#                               batcher,
#                               train_dataset,
#                               loss_func,
#                               statistics_func=error_func,
#                               val_dataset=val_dataset)

#     if log_bool():
#         logger.log("\n\nTESTING")
#     testing_parameters = {
#         'model_parameters':{},
#         'iterator_parameters':{
#             'indices_iterator_parameters':{'batch_size':15, 'random':True, 'iterations':200},
#             'batch_iterator_parameters':{},
#         },
#         'tester_parameters':{'print_every':10},
#     }
#     test_state = {'model_state':train_state['model_state']}
#     setup.test(testing_parameters,
#                batcher,
#                val_dataset,
#                loss_func,
#                statistics_func=error_func,
#                test_state=test_state)

if __name__ == '__main__':
    es = EmailSender(subject="pytt test", smtp_server='smtp.gmail.com', port=465, sender_email='jeredspython@gmail.com', receiver_email='jered.mcinerney@gmail.com')
    def onerror(e):
        logger.log(
            "Error sending email")
        if check_attachment_error(e):
            logger.log("Trying to send without attachment")
            es.send_email("email2")
        else:
            raise e
    es.send_email("starting pytt test", onerror=onerror, attachments=create_tensorboard_attachment_generator('/home/jered/Documents/projects/ehr-extraction-models/checkpoints/code_supervision_unfrozen2/tensorboard'))
    es = None
    nprocs = 2
    distributed_spawn_function = distributed_wrapper(spawn_function, nprocs, random_state=get_random_state())
    distributed_spawn_function(es)
    #setup(0,1)
    #spawn_function(es)
