# using vectorizer, handles batching from a (raw_dataset, indices) or (raw_input) -> batch with labels or batch without labels
#   (contains a batch_iterator (outputs batch with or without labels) which takes in a dataset, contains save, load, and print_training_batch_title functions)
#   (can debatch the output of the model to readable output)

class Batcher:
    def __init__(self, tokenizer):
        self.vectorizer = vectorizer

    def batch_from_dataset(self, raw_dataset, indices):
        pass

    def batch_from_raw(self, raw_input):
        pass

    def batch_iterator(self, dataset):
        return BatchIterator(self, dataset)

class BatchIterator:
    def __init__(self, batcher, dataset):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def training_batch_title(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        pass

def Instance:
    pass

def Batch:
    pass
