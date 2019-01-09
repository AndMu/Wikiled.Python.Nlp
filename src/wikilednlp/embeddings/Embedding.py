import heapq

import gensim
import numpy as np
from os import path
from sklearn import preprocessing

from wikilednlp.utilities.Utilities import Utilities
from wikilednlp.utilities import Constants
from wikilednlp.embeddings import logger


class Embedding(object):
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """

    def __init__(self, name, vocab, vectors):
        self.name = name
        self.vectors_raw = vectors
        self.vectors = vectors
        self.dim = vectors.shape[1]
        self.vocabulary = vocab

        self.word_index = {}
        self.index_word = {}
        index = Constants.EMBEDDING_START_INDEX
        for word in self.vocabulary:
            self.word_index[word] = index
            self.index_word[index] = word
            index += 1
        self.vectors = vectors

    def __getitem__(self, key):
        if self.oov(key):
            raise KeyError
        else:
            return self.represent(key)

    def __iter__(self):
        return self.vocabulary.__iter__()

    def __contains__(self, key):
        return not self.oov(key)

    @classmethod
    def load(cls, path, normalize=True, add_context=True):
        mat = np.load(path + "-w.npy")
        if add_context:
            mat += np.load(path + "-c.npy")
        iw = Utilities.load_pickle(path + "-vocab.pkl")
        return cls(mat, iw, normalize)

    def oov(self, w):
        return not (w in self.word_index)

    def represent(self, w):
        if w in self.word_index:
            return self.vectors[self.word_index[w], :]
        else:
            logger.info("OOV: %s", w)
            return np.zeros(self.dim)

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        sim = self.represent(w1).dot(self.represent(w2))
        return sim

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.vectors.dot(self.represent(w))
        return heapq.nlargest(n, zip(scores, self.vocabulary))


class SVDEmbedding(Embedding):
    """
    SVD embeddings.
    Enables controlling the weighted exponent of the eigenvalue matrix (eig).
    Context embeddings can be created with "transpose".
    """

    def __init__(self, file_path, eig=0.0):
        name = path.splitext(path.split(file_path)[-1])[0]
        ut = np.load(path + '-u.npy')
        s = np.load(path + '-s.npy')
        vocabfile = path + '-vocab.pkl'
        vocabulary = Utilities.load_pickle(vocabfile)
        if eig == 0.0:
            vectors = ut
        elif eig == 1.0:
            vectors = s * ut
        else:
            vectors = np.power(s, eig) * ut

        self.word_vector_table = {w: vectors[i] for i, w in enumerate(vocabulary)}
        super(SVDEmbedding, self).__init__('SVD_' + name, vocabulary, vectors)

        self.dim = self.vectors.shape[1]
        self.vectors_normalized = preprocessing.normalize(self.vectors, copy=True)


class GigaEmbedding(Embedding):
    def __init__(self, file_path, words=None):
        name = path.splitext(path.split(file_path)[-1])[0]
        seen = []
        self.word_vector_table = {}
        for line in Utilities.lines(file_path):
            split = line.split()
            word = split[0]
            try:
                if words is None or (word in words):
                    self.word_vector_table[word] = np.array(split[1:], dtype='float32')
                    seen.append(word)
            except ValueError:
                pass

        vectors = np.vstack(self.word_vector_table[w] for w in seen)
        super(GigaEmbedding, self).__init__('Giga_' + name, seen, vectors)
        self.vectors_normalized = preprocessing.normalize(self.vectors, copy=True)


class Word2VecEmbedding(Embedding):

    def __init__(self, model):
        vocabulary = list(model.vocab.keys())
        self.word_vector_table = {w: model[w] for w in vocabulary}
        super(Word2VecEmbedding, self).__init__('Word2Vec', vocabulary, model.syn0)
        model.init_sims()
        self.vectors_normalized = model.syn0norm
        self.model = model


class MainWord2VecEmbedding(Word2VecEmbedding):

    def __init__(self, file_name):
        logger.info('Loading word2vec [%s]', file_name)
        self.model = gensim.models.Word2Vec.load(file_name)
        super(MainWord2VecEmbedding, self).__init__(self.model.wv)    

    def save_word2vec_format(self, name, out_path, location=Constants.DATASETS):
        out_path = path.join(location, out_path)
        self.model.wv.save_word2vec_format(path.join(out_path, name + ".bin"), binary=True)