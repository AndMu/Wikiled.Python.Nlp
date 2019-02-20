import unittest

from os import path

import numpy as np

from wikilednlp.embeddings.Embedding import MainWord2VecEmbedding
from wikilednlp.embeddings.VectorManagers import Word2VecManager, EmbeddingManager
from wikilednlp.utilities import Constants


class Word2VecManagerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Constants.EMBEDDING_START_INDEX = 0
        cls.word2vec_zero = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)
        Constants.EMBEDDING_START_INDEX = 1
        cls.word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)
        Constants.use_special_symbols = True
        cls.word2vec_special = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)

    def test_special_construct(self):
        self.assertEqual(10002, self.word2vec_special.total_words)
        self.assertEqual(10002, len(self.word2vec_special.word_vectors))

    def test_construct(self):
        self.assertEqual(10000, self.word2vec.total_words)
        self.assertEqual(10000, len(self.word2vec.word_vectors))
        self.assertEqual("SemEval_min2", self.word2vec.name)
        self.assertEqual(500, self.word2vec.vector_size)
        self.assertEqual(0, self.word2vec.embedding_matrix[0][0])
        self.assertNotEquals(0, self.word2vec_zero.embedding_matrix[0][0])

    def test_construct_dataset(self):
        result = self.word2vec.construct_dataset(['good', 'bad'])
        self.assertEqual(2, len(result))
        self.assertEqual('good', result[0][0])
        self.assertEqual(500, len(result[0][1]))

    def test_get_matrix(self):
        data = np.array([[1, 0, 3], [10, 20, 30]])
        result = self.word2vec.get_matrix(data)
        self.assertEqual(2, len(result))


class EmbeddingManagerTest(unittest.TestCase):

    def test_word2vec_construct(self):
        embedding = MainWord2VecEmbedding(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'))
        word2vec = EmbeddingManager(embedding)
        self.assertEqual(303919, word2vec.total_words)
        self.assertEqual(303919, len(word2vec.word_vectors))
        self.assertEqual("Word2Vec", word2vec.name)
        self.assertEqual(500, word2vec.vector_size)
