# Pytt

NOTE: This package is actively under construction!

`pytt` is a package designed for training and testing (tt) pytorch models, offering a generalized, structured, and extensible way to batch data, load batches (using a pytorch dataloader as a backend with multithreading support), train model, checkpoint training, and test model.  It provides multi-gpu and delayed updates support, and any model trained with this is completely reproduceable because random states are a part of the checkpointing.

Some examples of options for customization:
* `pytt.batching.abstract_batcher.AbstractIndicesIterator` can be overridden and used to create an object that can be passed into the `pytt.batching.standard_batcher.Batcher.batch_iterator` method
* `pytt.batching.abstract_batcher.AbstractInstance` can be overridden and used in place of `pytt.batching.standard_batcher.StandardInstance` by overriding the `pytt.batching.standard_batcher.StandardBatcher.process_datapoint` method
* The `pytt.training.trainer.Trainer` can be extended by overriding the `process_batch` and `step` functions to use many different types of training proceedures.
