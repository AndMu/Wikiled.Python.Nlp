import unittest

import numpy as np

from wikilednlp.utilities.ClassConvertors import ClassConvertor


class ClassConvertorTests(unittest.TestCase):

    def test_make_dual(self):
        convertor = ClassConvertor("Test", {1: 1, 0: 0})
        self.assertEqual(1, convertor.is_supported(1))
        self.assertEqual(0, convertor.is_supported(0))

    def test_make_dual2(self):
        convertor = ClassConvertor("Test", {1: 2, 0: 1, -1: 0})
        self.assertEqual(2, convertor.is_supported(1))
        self.assertEqual(1, convertor.is_supported(0))
        self.assertEqual(0, convertor.is_supported(-1))

    def test_make_single(self):
        convertor = ClassConvertor("Test", {1: 1, 0: 0})
        y_test = np.array([[1, 0], [0, 1]])
        result = convertor.make_single(y_test)
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])
