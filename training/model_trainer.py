# trains model
# train function
#   takes in (checkpoint object, loss_func, optional error_func, optional custom step function, other options)
#   optionally saves checkpoint object
# contains a step function
#   takes in model, inputs and labels, loss_func, error_func
# Checkpoint object
#   contains a model, optimizer, batch_iterator, and optional val_iterator, optional_logfiles?
#   contains classmethods to load each independently from a file
#   contains classmethod to load checkpoint from folder
