import unittest

from os import path

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

    def test_construct(self):
        self.assertEquals(10000, self.word2vec.total_words)
        self.assertEquals(10000, len(self.word2vec.word_vectors))
        self.assertEquals("SemEval_min2", self.word2vec.name)
        self.assertEquals(500, self.word2vec.vector_size)
        self.assertEquals(0, self.word2vec.embedding_matrix[0][0])
        self.assertNotEquals(0, self.word2vec_zero.embedding_matrix[0][0])

    def construct_dataset(self):
        result = self.word2vec.construct_dataset(['good', 'bad'])
        self.assertEquals(2, len(result))
        self.assertEquals('good', result[0][0])
        self.assertEquals(500, len(result[0][1]))


class EmbeddingManagerTest(unittest.TestCase):

    def test_word2vec_construct(self):
        embedding = MainWord2VecEmbedding(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'))
        word2vec = EmbeddingManager(embedding)
        self.assertEquals(303919, word2vec.total_words)
        self.assertEquals(303919, len(word2vec.word_vectors))
        self.assertEquals("Word2Vec", word2vec.name)
        self.assertEquals(500, word2vec.vector_size)
