import unittest

from os import path

from nltk import TreebankWordTokenizer

from wikilednlp.embeddings.VectorSources import EmbeddingVecSource
from wikilednlp.utilities import Constants
from wikilednlp.embeddings.VectorManagers import WordVecManager
from wikilednlp.utilities.Lexicon import Lexicon


class EmbeddingVecSourceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Constants.set_special_tags(True)
        cls.word2vec = WordVecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)
        cls.lexicon = Lexicon(TreebankWordTokenizer())

    def setUp(self):
        self.source = EmbeddingVecSource(self.lexicon, self.word2vec)

    def test_get_vector_from_tokens(self):
        data_result = self.source.get_vector_from_tokens(('good', 'bad', 'xxxcx'))
        self.assertEqual(3, len(data_result))
        self.assertEqual(277, data_result[1])
        self.assertEqual(Constants.UNK_ID, data_result[2])
