import torch
from torch.optim import Adam
import torch.distributed as dist
from fairseq.legacy_distributed_data_parallel\
    import LegacyDistributedDataParallel as LDDP
from pytt.training.trainer import Trainer
from pytt.testing.tester import Tester
from pytt.batching.indices_iterator import init_indices_iterator

class Setup:
    @staticmethod
    def load_train_state(self, checkpoint_folder):
        model_state = torch.load(os.path.join(checkpoint_folder, 'model_state.tpkl'))
        optimizer_state = torch.load(os.path.join(checkpoint_folder, 'optimizer_state.pkl'))
        train_indices_iterator = read_from_file(os.path.join(checkpoint_folder, 'train_indices_iterator.pkl'))
        viifile = os.path.join(checkpoint_folder, 'val_indices_iterator.pkl')
        val_indices_iterator = read_from_file(viifile) if os.path.exists(viifile) else None
        return {
            'model_state':model_state,
            'optimizer_state':optimizer_state,
            'train_indices_iterator':train_indices_iterator,
            'val_indices_iterator':val_indices_iterator,
        }

    def __init__(self, model_class):
        self.model_class = model_class

    def train(self, training_parameters, batcher, train_dataset, loss_func, optimizer_class=Adam, trainer_class=Trainer, statistics_func=None, val_dataset=None, train_state=None):
        if train_state is not None:
            model_state, optimizer_state, train_indices_iterator, val_indices_iterator =\
                (train_state[s] if s in train_state.keys() else None
                 for s in (
                     'model_state',
                     'optimizer_state',
                     'train_indices_iterator',
                     'val_indices_iterator'))
        else:
            model_state, optimizer_state, train_indices_iterator, val_indices_iterator =\
                (None, None, None, None)
        model = self.get_model(training_parameters['model_parameters'],
                               state=model_state)
        optimizer = self.get_optimizer(optimizer_class,
                                       training_parameters['optimizer_parameters'],
                                       model,
                                       state=optimizer_state)
        batch_iterator = self.get_iterator(batcher,
                                           train_dataset,
                                           training_parameters['train_iterator_parameters'],
                                           state=train_indices_iterator)
        val_iterator = self.get_val_iterator(batcher,
                                             val_dataset,
                                             training_parameters['val_iterator_parameters'],
                                             batch_iterator,
                                             state=val_indices_iterator) if val_dataset is not None else None
        trainer = self.get_trainer(trainer_class, training_parameters['trainer_parameters'],
                                   model, optimizer, batch_iterator, val_iterator=val_iterator)
        trainer.train(loss_func, statistics_func=statistics_func)
        state_dict = model.module.state_dict() if isinstance(model, LDDP) else model.state_dict()
        return {
            'model_state':state_dict,
            'optimizer_state':optimizer.state_dict(),
            'train_indices_iterator':batch_iterator.indices_iterator,
            'val_indices_iterator':val_iterator.indices_iterator,
        }

    def get_model(self, arguments, state=None):
        model = self.model_class(**arguments)
        if state is not None:
            model.load_state_dict(state)
        if dist.is_initialized():
            model = LDDP(model, dist.get_world_size())
        return model

    def get_optimizer(self, optimizer_class, arguments, model, state=None):
        optimizer = optimizer_class(model.parameters(), **arguments)
        if state is not None:
            optimizer.load_state_dict(state)
        return optimizer

    def get_iterator(self, batcher, dataset, arguments, state=None):
        if state is None:
            if 'batch_size' not in arguments['indices_iterator_parameters'].keys():
                raise Exception
            indices_iterator = init_indices_iterator(len(dataset), **arguments['indices_iterator_parameters'])
        else:
            indices_iterator = state
        return batcher.batch_iterator(dataset, indices_iterator, **arguments['batch_iterator_parameters'])

    def get_val_iterator(self, batcher, dataset, arguments, batch_iterator, state=None):
        if 'iterations' in arguments['indices_iterator_parameters'].keys():
            raise Exception
        arguments['indices_iterator_parameters']['iterations'] = len(batch_iterator.indices_iterator)
        return self.get_iterator(batcher, dataset, arguments, state=state)

    def get_trainer(self, trainer_class, trainer_parameters, model, optimizer, batch_iterator, val_iterator=None):
        return trainer_class(model, optimizer, batch_iterator, val_iterator=val_iterator, **trainer_parameters)

    def test(self, testing_parameters, batcher, test_dataset, loss_func, tester_class=Tester, statistics_func=None, test_state=None):
        if test_state is not None:
            model_state, indices_iterator =\
                (test_state[s] if s in test_state.keys() else None
                 for s in ('model_state', 'indices_iterator'))
        else:
            model_state, indices_iterator = None, None
        model = self.get_model(testing_parameters['model_parameters'],
                               state=model_state)
        batch_iterator = self.get_iterator(batcher,
                                           test_dataset,
                                           testing_parameters['iterator_parameters'],
                                           state=indices_iterator)
        tester = self.get_tester(tester_class, testing_parameters['tester_parameters'], model, batch_iterator)
        tester.test(loss_func, statistics_func=statistics_func)

    def get_tester(self, tester_class, tester_parameters, model, batch_iterator):
        return tester_class(model, batch_iterator, **tester_parameters)
