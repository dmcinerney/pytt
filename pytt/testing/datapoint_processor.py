import torch

class DatapointProcessor:
    """
    Object used to process datapoints at test time
    """
    def __init__(self, model):
        self.model = model

    def process_batch(self, batch, test_func):
        # disable gradients
        with torch.autograd.no_grad():
            # run batch through the model
            outputs = self.model(**batch.get_observed())
            # run the test function on the outputs and batch targets
            stats = test_func(**outputs, **batch.get_target())
        return {**stats, "_batch_length":torch.tensor(len(batch))}
