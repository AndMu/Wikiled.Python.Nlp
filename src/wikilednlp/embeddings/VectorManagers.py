import abc
from enum import Enum

from nltk import PorterStemmer

from ..learning import logger
from ..utilities import Constants
from os import path
import numpy as np
import gensim

from ..utilities.TextHelper import TextHelper


class ManagerType(Enum):
        Binary = 1
        TextVectors = 2
        FastText = 3
        Word2Vec = 4


class BaseVecManager(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, total_words, word_index, index_word, word_vector_table, vector_size, word_vectors):
        self.name = name
        self.total_words = total_words
        self.word_index = word_index
        self.index_word = index_word
        self.word_vector_table = word_vector_table
        self.vector_size = vector_size
        self.word_vectors = word_vectors
        self.stemmer = PorterStemmer()
        self.emoticons = []
        self.hash_tags = []
        # prepare embedding matrix
        self.embedding_matrix = np.zeros((total_words + Constants.EMBEDDING_START_INDEX, vector_size))
        vector_unknown = np.zeros(vector_size)
        vector_unknown[0] = Constants.UNK_ID

        for word in word_vector_table.keys():
            embedding_vector = word_vector_table.get(word)
            if embedding_vector is None:
                # words not found in embedding
                embedding_vector = vector_unknown

            i = word_index[word]
            self.embedding_matrix[i] = embedding_vector

            if TextHelper.is_emoticon(word):
                self.emoticons.append(word)
            if TextHelper.is_hash(word):
                self.hash_tags.append(word)

        logger.debug("Initialized")

    def get_matrix(self, data):
        logger.debug("Getting matrix")
        embedding = np.zeros((data.shape[0], data.shape[1], self.vector_size))
        document_id = 0
        for document in data:
            vec = np.array([self.embedding_matrix[d] for d in document])
            embedding[document_id] = vec
            document_id += 1
        logger.debug("Getting matrix Done!")
        return embedding

    def get_vocabulary(self):
        return self.word_index

    def get_tokens(self, tokens):
        result = []
        for word in tokens:
            if word in self.word_index:
                result.append(self.word_vector_table[word])
            else:
                stemmed = self.stemmer.stem(word)
                if stemmed in self.word_index:
                    result.append(self.word_vector_table[stemmed])
        return result

    def construct_dataset(self, words):
        vectors = []
        for word in words:
            item = word.lower()
            vector = self.word_vector_table[item]
            vectors.append((item, vector))
        return vectors


class WordVecManager(BaseVecManager):

    def __init__(self, file_name, model_type=ManagerType.Word2Vec, vocab_size=10000):
        name = path.splitext(path.split(file_name)[-1])[0]
        self.model_type = model_type
        w2vModel = self.construct(file_name)
        logger.info('Sorting words')
        sorted_list = sorted(w2vModel.wv.vocab.items(), key=lambda t: t[1].count, reverse=True)[0:vocab_size]
        total_words = len(sorted_list)

        word_index = {}
        index_word = {}
        word_vector_table = {}
        vectors = []
        vector_size = w2vModel.vector_size

        if Constants.EMBEDDING_START_INDEX <= 0:
            raise ValueError('Embedding index is too low')

        index = 0

        def add_vector(id, word, word_vector):
            nonlocal index
            nonlocal vectors
            nonlocal word_index
            nonlocal index_word
            nonlocal word_vector_table

            word_index[word] = id
            index_word[id] = word
            word_vector_table[word] = word_vector
            vectors.append(word_vector)
            index += 1

        vector = np.zeros(vector_size)
        add_vector(Constants.PAD_ID, Constants.PAD, vector)

        if Constants.use_special_symbols:
            logger.info('Inserting special Symbols')

            vector = np.zeros(vector_size)
            vector[0] = Constants.START_ID
            add_vector(Constants.START_ID, Constants.START, vector)

            vector = np.zeros(vector_size)
            vector[0] = Constants.END_ID
            add_vector(Constants.END_ID, Constants.END, vector)

            vector = np.zeros(vector_size)
            vector[0] = Constants.UNK_ID
            add_vector(Constants.UNK_ID, Constants.UNK, vector)

        for wordKey in sorted_list:
            word = wordKey[0]
            if len(word) == 0:
                continue

            vector = w2vModel.wv[word]
            add_vector(index, word, vector)

        word_vectors = np.array(vectors)
        self.w2vModel = w2vModel
        super(WordVecManager, self).__init__(name, total_words, word_index, index_word, word_vector_table, vector_size,
                                             word_vectors)

    def construct(self, file_name):
        logger.info('Loading Word2Vec...')
        if self.model_type == ManagerType.Binary:
            logger.info('Loading binary version')
            return gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True)
        elif self.model_type == ManagerType.TextVectors:
            logger.info('Loading text version')
            return gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=False)
        elif self.model_type == ManagerType.FastText:
            logger.info('Loading fasttext version')
            return gensim.models.FastText.load(file_name)
        return gensim.models.Word2Vec.load(file_name)


# takes generic embedding class to load vectors
class EmbeddingManager(BaseVecManager):
    def __init__(self, embeddings):
        total_words = len(embeddings.vocabulary)
        word_index = embeddings.word_index
        index_word = embeddings.index_word
        vector_size = embeddings.dim
        word_vector_table = embeddings.word_vector_table
        word_vectors = embeddings.vectors
        super(EmbeddingManager, self).__init__(embeddings.name, total_words, word_index, index_word, word_vector_table,
                                               vector_size, word_vectors)
