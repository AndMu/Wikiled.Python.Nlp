import unittest

from ddt import ddt, data
from nltk import TreebankWordTokenizer
from os import path

from wikilednlp.embeddings.VectorManagers import Word2VecManager
from wikilednlp.embeddings.VectorSources import EmbeddingVecSource
from wikilednlp.utilities import Constants
from wikilednlp.utilities.DataLoaders import ImdbDataLoader
from wikilednlp.utilities.Lexicon import Lexicon
from wikilednlp.utilities.ClassConvertors import ClassConvertor

@ddt
class ImdbDataLoaderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)
        cls.source = EmbeddingVecSource(lexicon, word2vec)
        cls.convertor = ClassConvertor("Binary", {"0": 0, "1": 1})

    def test_get_data(self):
        loader = ImdbDataLoader(self.source, self.convertor, root=path.join(Constants.DATASETS, 'Test'))
        # delete records, so no clash between dual and single
        record = loader.get_data('train', delete=True)
        self.assertEqual(20, len(record.names))
        self.assertEqual(20, len(record.x_data))
        self.assertEqual(20, len(record.y_data))
        self.assertEqual(20, len(record.block_ids))
        # Test loading
        record = loader.get_data('train', delete=False)
        self.assertEqual(20, len(record.names))
        self.assertEqual(20, len(record.x_data))
        self.assertEqual(20, len(record.y_data))
        self.assertEqual(20, len(record.block_ids))

    def test_get_single_data(self):
        loader = ImdbDataLoader(self.source, self.convertor, root=path.join(Constants.DATASETS, 'Test'))
        # delete records, so no clash between dual and single
        record = loader.get_data('train/pos', delete=True, class_iter=False)
        self.assertEqual(10, len(record.names))
        self.assertEqual(10, len(record.x_data))
        self.assertEqual(10, len(record.y_data))
        self.assertEqual(10, len(record.block_ids))
        # Test loading
        record = loader.get_data('train/pos', delete=False, class_iter=False)
        self.assertEqual(10, len(record.names))
        self.assertEqual(10, len(record.x_data))
        self.assertEqual(10, len(record.y_data))
        self.assertEqual(10, len(record.block_ids))

    @data(True, False)
    def test_unknown_data(self, class_iter):
        loader = ImdbDataLoader(self.source, self.convertor, root=path.join(Constants.DATASETS, 'Test'))
        self.assertRaises(Exception, loader.get_data, 'xxx', delete=True, class_iter=class_iter)
