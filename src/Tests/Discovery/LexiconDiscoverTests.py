import unittest
from os import path

from wikilednlp.Discovery import seeds
from wikilednlp.Discovery.LexiconDiscover import LexiconDiscover
from wikilednlp.embeddings.VectorManagers import WordVecManager
from wikilednlp.utilities import Constants


class LexiconDiscoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.word2vec = WordVecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=100000)

    def setUp(self):
        self.discover = LexiconDiscover(self.word2vec, seeds.turney_seeds)

    def test_discover(self):
        result = self.discover.discover()
        self.assertGreater(len(result.positive), 100)
        self.assertGreater(len(result.negative), 100)

    def test_discover_construct(self):
        result = self.discover.discover()
        dataset = self.discover.construct(result)
        dataset.to_csv('words.csv', index=False, header=False, encoding='UTF8')
        self.assertGreater(len(dataset), 300)
