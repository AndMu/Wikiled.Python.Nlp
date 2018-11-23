import unittest

from os import path

from nltk import TreebankWordTokenizer

from WikiledNlp.embeddings.VectorSources import EmbeddingVecSource
from WikiledNlp.utilities import Constants
from WikiledNlp.embeddings.VectorManagers import Word2VecManager
from WikiledNlp.utilities.Lexicon import Lexicon


class EmbeddingVecSourceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), top=10000)
        cls.lexicon = Lexicon(TreebankWordTokenizer())

    def setUp(self):
        self.source = EmbeddingVecSource(self.lexicon, self.word2vec)

    def test_get_vector_from_tokens(self):
        data_result = self.source.get_vector_from_tokens(('good', 'bad'))
        self.assertEquals(2, len(data_result))
        self.assertEquals(274, data_result[1])
