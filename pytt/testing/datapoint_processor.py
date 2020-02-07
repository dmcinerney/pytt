import torch

class DatapointProcessor:
    """
    Object used to process datapoints at test time
    """
    def __init__(self, model, postprocessor):
        self.model = model
        self.postprocessor = postprocessor

    def process_batch(self, batch):
        # disable gradients
        with torch.autograd.no_grad():
            # run batch through the model
            outputs = self.model(**batch.get_observed())
            # run the test function on the outputs and batch targets
            kwargs = {**outputs, **batch.get_target()}
            # ignore loss
            return self.postprocessor.output_batch(batch, kwargs)[1]
