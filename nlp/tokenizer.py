import torch

class Tokenizer:
    """
    Handles all tokenization
    """
    def __init__(self, vocab):
        """
        Initializes tokenizer using input vocabulary in the form of an iterable object
        """
        self._token2id = {}
        self._id2token = {}
        for i,token in enumerate(vocab):
            self._token2id[token] = i
            self._id2token[i] = token

    def token2id(self, token, oov_token2id={}):
        """
        Converts token to index, using negative indices for oov tokens,
        optionally working off of and adding to an already existing oov token to index mapping
        """
        if token in self._token2id.keys():
            return self._token2id[token]
        else:
            if token not in oov_token2id.keys():
                oov_token2id[token] = len(oov_token2id)
            return -1-oov_token2id[token]

    def id2token(self, index, oov_id2token={}):
        """
        Converts index to token, optionally taking into account an additional
        oov index to token mapping for negative indices
        """
        if index >= len(self._id2token):
            raise Exception
        if i >= 0:
            return self._id2token[i]
        else:
            if i < -len(oov_id2token):
                return 'oov'
            else:
                return oov_id2token[-i-1]

    def tokens2tensor(self, tokens, oov_token2id={}):
        """
        Returns a tensor mapped from the input tokens using self.token2id
        and an oov_token2id mapping (see self.token2id)
        """
        oov_token2id = dict(oov_token2id)
        tensor = torch.zeros(len(tokens)).long()
        for i,token in enumerate(tokens):
            tensor[i] = self.token2id(token, oov_token2id=oov_token2id)
        return tensor, oov_token2id

    def tensor2tokens(self, tensor, oov_id2token={}):
        """
        Returns a generator of tokens mapped from the input tensor using self.id2token
        """
        return (self.id2token(i) for i in tensor)
