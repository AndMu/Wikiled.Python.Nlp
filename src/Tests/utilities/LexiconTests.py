import unittest
from os import path

from nltk import TreebankWordTokenizer

from wikilednlp.embeddings.VectorManagers import WordVecManager
from wikilednlp.utilities import Constants
from wikilednlp.utilities.Lexicon import Lexicon


class LexiconTests(unittest.TestCase):

    def setUp(self):
        self.lexicon = Lexicon(TreebankWordTokenizer())

    def test_word_tokenize(self):
        tokens = self.lexicon.word_tokenize('My sample text')
        self.assertEqual(3, len(tokens))

    def test_review_to_wordlist(self):
        tokens = self.lexicon.review_to_wordlist('My the sample text')
        self.assertEqual(4, len(tokens))
        self.lexicon.remove_stopwords = True
        tokens = self.lexicon.review_to_wordlist('My the sample text')
        self.assertEqual(2, len(tokens))
        Constants.use_special_symbols = True
        tokens = self.lexicon.review_to_wordlist('My the sample text')
        self.assertEqual(4, len(tokens))

    def test_review_to_sentences(self):
        tokens = self.lexicon.review_to_sentences('My the sample text')
        self.assertEqual(1, len(list(tokens)))


class Word2VecManagerTests(unittest.TestCase):

    def setUp(self):
        self.lexicon = WordVecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'))

    def test_construction(self):
        self.assertEqual('SemEval_min2', self.lexicon.name)
