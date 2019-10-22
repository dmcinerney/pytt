import torch

class DatapointProcessor:
    """
    Object used to process datapoints at test time
    """
    def __init__(self, model, batch_info):
        self.model = model

    def process_batch(self, batch):
        # disable gradients
        with torch.autograd.no_grad():
            # run batch through the model
            outputs = self.model(**batch.get_observed())
            # run the test function on the outputs and batch targets
        kwargs = {**outputs, **batch.get_target()}
        return self.batch_info_class(kwargs, len(batch))
