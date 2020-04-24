import abc
import csv
from enum import Enum

from nltk import PorterStemmer

from ..learning import logger
from ..utilities import Constants
from os import path
import io
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
        self.embedding_matrix = np.array(word_vectors)

        for word in word_vector_table.keys():
            if TextHelper.is_emoticon(word):
                self.emoticons.append(word)
            if TextHelper.is_hash(word):
                self.hash_tags.append(word)

        logger.debug("Initialized")

    def save_dictionary(self, file_path):
        headers = ['Id', 'Word']
        with open(file_path, 'w', encoding='utf-8') as dictionary_file:
            writer = csv.DictWriter(dictionary_file, delimiter='\t', lineterminator='\n', fieldnames=headers)
            writer.writeheader()
            for key, value in self.word_index.items():
                writer.writerow(
                    {
                        'Id': key,
                        'Word': value
                    })


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
        _w2v_model = self._construct(file_name)
        logger.info('Sorting words')
        sorted_list = sorted(_w2v_model.wv.vocab.items(), key=lambda t: t[1].count, reverse=True)[0:vocab_size]
        total_words = len(sorted_list)

        word_index = {}
        index_word = {}
        word_vector_table = {}
        vectors = []
        vector_size = _w2v_model.vector_size

        if Constants.EMBEDDING_START_INDEX <= 0:
            raise ValueError('Embedding index is too low')

        index = 0

        def add_vector(word_local, word_vector):
            nonlocal index
            nonlocal vectors
            nonlocal word_index
            nonlocal index_word
            nonlocal word_vector_table

            word_index[word_local] = index
            index_word[index] = word_local
            word_vector_table[word_local] = word_vector
            vectors.append(word_vector)
            index += 1

        # add zero pad vector
        vector = np.zeros(vector_size)
        add_vector(Constants.PAD, vector)

        if Constants.use_special_symbols:
            logger.info('Inserting special Symbols')

            vector = np.zeros(vector_size)
            vector[Constants.START_ID] = 1
            add_vector(Constants.START, vector)

            vector = np.zeros(vector_size)
            vector[Constants.END_ID] = 1
            add_vector(Constants.END, vector)

            vector = np.zeros(vector_size)
            vector[Constants.UNK_ID] = 1
            add_vector(Constants.UNK, vector)

        for wordKey in sorted_list:
            word = wordKey[0]
            if len(word) == 0:
                continue

            vector = _w2v_model.wv[word]
            add_vector(word, vector)

        word_vectors = np.array(vectors)
        self.w2vModel = _w2v_model
        super(WordVecManager, self).__init__(name, total_words, word_index, index_word, word_vector_table, vector_size,
                                             word_vectors)

    def save_index(self, file_name):
        with io.open(file_name, 'wt', encoding='utf8') as csv_file:
            for key, value in self.index_word.items():
                csv_file.write(f'{key}\t{value}\n')

    def _construct(self, file_name):
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

