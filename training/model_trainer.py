# trains model
# train function
#   takes in (checkpoint object, loss_func, optional error_func, optional custom step function, other options)
#   optionally saves checkpoint object
# contains a step function
#   takes in model, inputs and labels, loss_func, error_func
# Checkpoint object
#   contains a model, optimizer, batch_iterator, and optional val_iterator
#   contains classmethods to load each independently from a file
#   contains classmethod to load checkpoint from folder

def train(checkpoint, loss_func, error_func=None, step_func=defualt_step_func, grad_mod=None, nan_to_none=False):
    if no_nan_grad:
        for p in checkpoint.model.parameters():
            p.register_hook(nan_to_num_hook)
    for batch in checkpoint.batch_iterator:
        step_func(checkpoint.model, batch, loss_func, error_func=error_func, training=True, grad_mod=grad_mod)
        if checkpoint.val_iterator is not None:
             = next(val_iterator)

def default_step_func(model, batch, loss_func, error_func=None, training=False, grad_mod=None):
    with torch.set_grad_enabled(training):
        outputs = self.model(instance)
        loss = self.loss_function(**outputs)
    with torch.autograd.no_grad():
        error = self.error_function(**outputs)
    if training:
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_mod is not None:
            self.grad_mod(self.model.parameters())
        self.optimizer.step()
    loss_value = loss.item()
    if error is not None:
        error_value = error.item()
    else:
        error_value = None
    return loss_value, error_value

class Checkpoint:
    def __init__(self, model, optimizer, batch_iterator, val_iterator=None):
        self.model = model
        self.optimizer = optimizer
