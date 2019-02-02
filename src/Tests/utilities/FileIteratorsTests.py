import unittest

from os import path

from ddt import ddt, data, unpack
from mock import patch
from nltk import TreebankWordTokenizer

from wikilednlp.embeddings.VectorManagers import Word2VecManager
from wikilednlp.utilities import Constants
from wikilednlp.utilities.FileIterators import ClassDataIterator, SemEvalDataIterator
from wikilednlp.utilities.Lexicon import Lexicon
from wikilednlp.utilities.ClassConvertors import ClassConvertor
from wikilednlp.embeddings.VectorSources import EmbeddingVecSource


class DataIteratorTests(unittest.TestCase):

    def setUp(self):
        with patch('wikilednlp.embeddings.VectorSources.EmbeddingVecSource') as mock:
            instance = mock.instance
            instance.word2vec.name = 'name'
            self.iterator = ClassDataIterator(instance, 'root', 'test')

    def test_bin_location(self):
        self.assertEqual('C:/Temp/Sentiment\\bin\\root\\test\\name\\sentence', self.iterator.bin_location)


@ddt
class SemEvalDataIteratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/Imdb_min2.bin'), vocab_size=10000)
        cls.source = EmbeddingVecSource(lexicon, word2vec)

    def setUp(self):
        self.source.use_sentence = False

    @data([2, 8047, 5690], [3, 14885, 5690])
    @unpack
    def test_parsing(self, num_class, expected, expected_pos):
        covertor = ClassConvertor("Binary", {"positive": 1, "negative": 0})
        if num_class == 3:
            covertor = ClassConvertor("Three", {"positive": 2, "negative": 0, "neutral": 1})
        iterator = SemEvalDataIterator(self.source, path.join(Constants.DATASETS, 'Test'), 'SemEval', covertor)
        iterator.delete_cache()
        record = iterator.get_data()
        x, y = record.get_vectors()
        self.assertEqual(expected, len(x))
        record = iterator.get_data()
        x, y = record.get_vectors()
        self.assertEqual(expected, len(x))
        class_id = num_class - 1
        self.assertEqual(expected_pos, sum(y == class_id))

    @data([2, 915, 575], [3, 1654, 739])
    @unpack
    def test_parsing_file(self, num_class, expected, expected_pos):

        covertor = ClassConvertor("Binary", {"positive": 1, "negative": 0})
        if num_class == 3:
            covertor = ClassConvertor("Three", {"positive": 2, "negative": 0, "neutral": 1})

        iterator = SemEvalDataIterator(self.source, path.join(Constants.DATASETS, 'Test'),
                                       'SemEval/twitter-2013dev-A.txt', covertor)
        iterator.delete_cache()
        record = iterator.get_data()
        x, y = record.get_vectors()
        self.assertEqual(expected, len(x))
        record = iterator.get_data()
        x, y = record.get_vectors()
        self.assertEqual(expected, len(x))
        self.assertEqual(expected_pos, sum(y == 1))

    def test_parsing_multiclass_file(self):

        covertor = ClassConvertor("Multi ", {"-2": 0, "-1": 0, "0": 1, "1": 2, "2": 2})
        iterator = SemEvalDataIterator(self.source, path.join(Constants.DATASETS, 'Test'),
                                       'SemEval/twitter-2016devtest-CE.out', covertor)
        iterator.delete_cache()
        record = iterator.get_data()
        x, y = record.get_vectors()
        self.assertEqual(2000, len(x))
        self.assertEqual(264, sum(y == 0))
        self.assertEqual(583, sum(y == 1))
        self.assertEqual(1153, sum(y == 2))

    def test_parsing_multiclass_sentences(self):
        self.source.use_sentence = True
        covertor = ClassConvertor("Multi ", {"-2": 0, "-1": 0, "0": 1, "1": 2, "2": 2})
        iterator = SemEvalDataIterator(self.source, path.join(Constants.DATASETS, 'Test'),
                                       'SemEval/twitter-2016devtest-CE.out', covertor)
        iterator.delete_cache()
        record = iterator.get_data()
        x, y = record.get_vectors()
        self.assertEqual(3686, len(x))
        self.assertEqual(477, sum(y == 0))
        self.assertEqual(1028, sum(y == 1))
        self.assertEqual(2181, sum(y == 2))
