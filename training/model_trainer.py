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

def train(checkpoint, loss_func, error_func=None, step_func=defualt_step_func, grad_mod=None, nan_to_zero=False, verbose_every=1, write_every=1):
    """
    Takes in (
        checkpoint - object containing model, optimizer, and batch_iterator, and optional val_iterator,
        loss_func,
        optional error_func,
        optional custom step function (defualt default_step_func),
        optional grad_mod function,
        optional nan_to_zero (default False) - an option to convert nan grads to zero,
        optional verbose_every (default 1) - an option to specify how often to print training info,
        optional write_every (default 1) - an option to specify how often to save the checkpoint,
    ) and trains model
    """
    # allow model to set all nan gradients to zero
    if no_nan_grad:
        for p in checkpoint.model.parameters():
            p.register_hook(nan_to_num_hook)
    for batch in checkpoint.batch_iterator:
        step_func(checkpoint.model, batch, loss_func, error_func=error_func, training=True, grad_mod=grad_mod)
        if checkpoint.val_iterator is not None:
            val_batch = next(val_iterator)
            step_func(checkpoint.model, val_batch, loss_func, error_func=error_func, training=False, grad_mod=grad_mod)

def nan_to_zero_hook(grad):
    """
    A backward hook function that sets all nan grads to zero
    """
    new_grad = grad.clone()
    if (new_grad != new_grad).any():
        print("Warning: NaN encountered!")
        print(grad)
    new_grad[new_grad != new_grad] = 0
    return new_grad

def default_step_func(model, batch, loss_func, error_func=None, training=False, grad_mod=None):
    """
    Takes in (
        model,
        batch,
        loss_func,
        optional error_func,
        optional training (default False) - whether or not to enable grads,
        optional grad_mod - a function to alter the grads before taking a step,
    ) and steps through the model, returning loss and error (None if no error is specified)
    """
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
    """
    Checkpoint object containing model, optimizer, batch_iterator and val_iterator
    with saving and loading capabilities
    """
    def __init__(self, model, optimizer, batch_iterator, val_iterator=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_iterator = batch_iterator
        self.val_iterator = val_iterator
