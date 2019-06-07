# trains model
# train function
#   takes in (checkpoint object, loss_func, optional error_func, optional custom
#       step function, other options)
#   optionally saves checkpoint object
# contains a step function
#   takes in model, inputs and labels, loss_func, error_func
# Checkpoint object
#   contains a model, optimizer, batch_iterator, and optional val_iterator
#   contains classmethods to load each independently from a file
#   contains classmethod to load checkpoint from folder


def train(checkpoint, loss_func, error_func=None, step_func=defualt_step_func,
          grad_mod=None, verbose_every=1, write_every=1):
    """
    Takes in (
        checkpoint - object containing model, optimizer, and batch_iterator, and
            optional val_iterator,
        loss_func,
        optional error_func,
        optional custom step function (defualt default_step_func),
        optional grad_mod function,
        optional verbose_every (default 1) - an option to specify how often to
            print training info,
        optional write_every (default 1) - an option to specify how often to
            save the checkpoint,
    ) and trains model
    """
    for batch in checkpoint.batch_iterator:
        step_info = {"iterator_info":checkpoint.batch_iterator.iterator_info()}
        train_loss, train_error = step_func(checkpoint.model, batch, loss_func,
                                            error_func=error_func,
                                            enable_grad=True, grad_mod=grad_mod)
        step_info.update({
            "train_loss":train_loss,
            "train_error":train_error,
        })
        if step_info["iterator_info"]["batches_seen"] % write_every == 0:
            if checkpoint.val_iterator is not None:
                val_batch = next(val_iterator)
                val_loss, val_error = step_func(checkpoint.model, val_batch,
                                                loss_func,
                                                error_func=error_func,
                                                enable_grad=False,
                                                grad_mod=grad_mod)
                step_info.update({
                    "val_loss":train_loss,
                    "val_error":train_error,
                })
        checkpoint.step_info(step_info)


def default_step_func(model, batch, loss_func, error_func=None,
                      enable_grad=True, grad_mod=None):
    """
    Takes in (
        model,
        batch,
        loss_func,
        optional error_func,
        optional training (default False) - whether or not to enable grads,
        optional grad_mod - a function to alter the grads before taking a step,
    ) and steps through the model, returning loss and error (None if no error is
    specified)
    """
    with torch.set_grad_enabled(enable_grad):
        outputs = model(batch)
        loss = loss_func(**outputs)
    with torch.autograd.no_grad():
        error = error_func(**outputs) if error_func is not None else None
    if training:
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_mod is not None:
            self.grad_mod(self.model.parameters())
        self.optimizer.step()
    loss_value = loss.item()
    error_value = error.item() if error is not None else None
    return loss_value, error_value


class Checkpoint:
    """
    Checkpoint object containing model, optimizer, batch_iterator and
    val_iterator with saving and loading capabilities
    """
    @classmethod
    def load(cls, folder):
        raise NotImplementedError

    def __init__(self, model, optimizer, batch_iterator, val_iterator=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_iterator = batch_iterator
        self.val_iterator = val_iterator

    def step_info(self, info):
        raise NotImplementedError

    def save(self, folder):
        raise NotImplementedError
