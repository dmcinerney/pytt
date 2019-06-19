from pytt.preprocessing.raw_dataset import RawDataset
import pandas as pd
# from gensim.models import Word2Vec
# from gensim.corpora import Dictionary

class SummarizationDataset(RawDataset):
    def __init__(self, filename):
        self.df = pd.read_json(filename, lines=True, compression='gzip')

class TextIterator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self

    def __next__(self):
        for i,row in self.dataset.df.iterrows():
            import pdb; pdb.set_trace()
            yield row.text
            yield row.summary

# create gensim word2vec model using dataset
# def train_gensim_word2vec_model(data_file, word2vec_file, embedding_dim):
#     print("reading data file")
#     document_iterator = SummarizationDataset(data_file).text_iterator()
#     print("creating dictionary")
#     word2vec_model = Word2Vec(document_iterator, size=embedding_dim, window=5,
#                               min_count=83, workers=4)
#     word2vec_model.save(word2vec_file)
    
# create gensim dictionary using dataset
# def save_gensim_dictionary(data_file, dictionary_file):
#     print("reading data file")
#     document_iterator = SummarizationDataset(
#         data_file, aspect_file=aspect_file).text_iterator()
#     print("creating dictionary")
#     dictionary = Dictionary(document_iterator, prune_at=50000)
#     dictionary.save(dictionary_file)

# load in vocabulary file
def load_vocab(filename, max_size):
    with open(filename, 'rb') as f:
        for i,line in enumerate(f):
            if i >= max_size:
                break
            token, _ = line.split()
            yield token.decode()
