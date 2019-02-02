import unittest

from ddt import ddt, data
from nltk import TreebankWordTokenizer
from os import path

from wikilednlp.embeddings.VectorManagers import Word2VecManager
from wikilednlp.embeddings.VectorSources import EmbeddingVecSource
from wikilednlp.utilities import Constants
from wikilednlp.utilities.DataLoaders import ImdbDataLoader
from wikilednlp.utilities.Lexicon import Lexicon

@ddt
class ImdbDataLoaderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)
        cls.source = EmbeddingVecSource(lexicon, word2vec)

    def test_get_data(self):
        loader = ImdbDataLoader(self.source, root=path.join(Constants.DATASETS, 'Test'))
        # delete records, so no clash between dual and single
        record = loader.get_data('train', delete=True)
        self.assertEqual(20, len(record.records))
        # Test loading
        record = loader.get_data('train', delete=False)
        self.assertEqual(20, len(record.records))
        x, y = record.get_vectors()
        self.assertEqual(20, len(x))
        self.assertEqual(20, len(y))

    def test_get_single_data(self):
        loader = ImdbDataLoader(self.source, root=path.join(Constants.DATASETS, 'Test'))
        # delete records, so no clash between dual and single
        record = loader.get_data('train/pos', delete=True, class_iter=False)
        self.assertEqual(10, len(record.records))
        # Test loading
        record = loader.get_data('train/pos', delete=False, class_iter=False)
        self.assertEqual(10, len(record.records))
        x, y = record.get_vectors()
        self.assertEqual(10, len(x))
        self.assertEqual(10, len(y))

    @data(True, False)
    def test_unknown_data(self, class_iter):
        loader = ImdbDataLoader(self.source, root=path.join(Constants.DATASETS, 'Test'))
        self.assertRaises(Exception, loader.get_data, 'xxx', delete=True, class_iter=class_iter)
