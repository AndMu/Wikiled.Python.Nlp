import os
import unittest

from os import path

import numpy as np

from wikilednlp.embeddings.VectorManagers import WordVecManager
from wikilednlp.utilities import Constants


class Word2VecManagerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Constants.set_special_tags(False)
        cls.word2vec = WordVecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)
        Constants.set_special_tags(True)
        cls.word2vec_special = WordVecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), vocab_size=10000)

    def test_special_construct(self):
        self.assertEqual(10000, self.word2vec_special.total_words)
        self.assertEqual(10004, len(self.word2vec_special.word_vectors))
        padding = self.word2vec_special.word_vector_table[Constants.PAD]
        self.assertEqual(0, padding[0])
        start = self.word2vec_special.word_vector_table[Constants.START]
        self.assertEqual(1, start[Constants.START_ID])
        end = self.word2vec_special.word_vector_table[Constants.END]
        self.assertEqual(1, end[Constants.END_ID])
        unk = self.word2vec_special.word_vector_table[Constants.UNK]
        self.assertEqual(1, unk[Constants.UNK_ID])

    def test_construct(self):
        self.assertEqual(10000, self.word2vec.total_words)
        self.assertEqual(10001, len(self.word2vec.word_vectors))
        self.assertEqual("SemEval_min2", self.word2vec.name)
        self.assertEqual(500, self.word2vec.vector_size)
        self.assertEqual(0, self.word2vec.embedding_matrix[0][0])

    def test_construct(self):
        file_name = 'dictionary.dic'
        if os.path.exists(file_name):
            os.remove(file_name)
        self.word2vec.save_dictionary(file_name)
        with open(file_name) as f:
            content = f.readlines()
            self.assertEqual(10002, len(content))

    def test_construct_dataset(self):
        result = self.word2vec.construct_dataset(['good', 'bad'])
        self.assertEqual(2, len(result))
        self.assertEqual('good', result[0][0])
        self.assertEqual(500, len(result[0][1]))

    def test_get_matrix(self):
        data = np.array([[1, 0, 3], [10, 20, 30]])
        result = self.word2vec.get_matrix(data)
        self.assertEqual(2, len(result))
