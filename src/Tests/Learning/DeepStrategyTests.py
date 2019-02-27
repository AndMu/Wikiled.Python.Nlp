import unittest

from keras.preprocessing import sequence
from nltk import TreebankWordTokenizer
from os import path

from wikilednlp.embeddings.VectorSources import EmbeddingVecSource
from wikilednlp.embeddings.VectorManagers import WordVecManager
from wikilednlp.learning.DeepLearning import CnnSentiment
from wikilednlp.utilities import Constants
from wikilednlp.utilities.DataLoaders import ImdbDataLoader
from wikilednlp.utilities.Lexicon import Lexicon


class WeightsLSTMTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec = WordVecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)
        source = EmbeddingVecSource(lexicon, word2vec)
        cls.loader = ImdbDataLoader(source, root=path.join(Constants.DATASETS, 'test'))
        pass

    def test_acceptance(self):
        first = CnnSentiment(self.loader, 'AcceptanceTest', 500)
        x, y = self.loader.get_data('train', delete=True).get_vectors()
        train_x = sequence.pad_sequences(x, maxlen=500)
        first.fit(train_x, y)
        x, y = self.loader.get_data('test', delete=True).get_vectors()
        test_x = sequence.pad_sequences(x, maxlen=500)
        y = first.predict(test_x)
        self.assertEqual(20, len(y))
        self.assertGreater(sum(y > 0.5), 0)
        self.assertGreater(sum(y < 0.5), 0)
