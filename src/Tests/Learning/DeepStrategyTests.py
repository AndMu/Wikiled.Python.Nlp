import unittest

from keras.preprocessing import sequence
from nltk import TreebankWordTokenizer
from os import path

from wikilednlp.embeddings.VectorSources import EmbeddingVecSource
from wikilednlp.utilities.ClassConvertors import ClassConvertor
from wikilednlp.embeddings.VectorManagers import Word2VecManager
from wikilednlp.learning.DeepLearning import CnnSentiment
from wikilednlp.utilities import Constants
from wikilednlp.utilities.DataLoaders import ImdbDataLoader
from wikilednlp.utilities.Lexicon import Lexicon


class WeightsLSTMTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)
        source = EmbeddingVecSource(lexicon, word2vec)
        class_convertor = ClassConvertor("Binary", {"0": 0, "1": 1})
        cls.loader = ImdbDataLoader(source, class_convertor, root=path.join(Constants.DATASETS, 'test'))
        pass

    def test_acceptance(self):
        first = CnnSentiment(self.loader, 'AcceptanceTest', 500, vocab_size=10000)
        data = self.loader.get_data('train', delete=True)
        train_x = sequence.pad_sequences(data.x_data, maxlen=500)
        first.fit(train_x, data.y_data)
        data = self.loader.get_data('test', delete=True)
        test_x = sequence.pad_sequences(data.x_data, maxlen=500)
        y = first.predict(test_x)
        self.assertEqual(20, len(y))
        self.assertGreater(sum(y > 0.5), 0)
        self.assertGreater(sum(y < 0.5), 0)
