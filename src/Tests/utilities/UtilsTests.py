import unittest

import numpy as np

from wikilednlp.utilities.ClassConvertors import ClassConvertor
from wikilednlp.utilities.Utilities import Utilities


class UtilsTests(unittest.TestCase):

    def test_make_binary_dual(self):
        y_test = np.array([[0.61, 0.1], [0.1, 0.52]])
        result = Utilities.make_single_dimension(y_test)
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])

    def test_make_binary(self):
        y_test = np.array([0.1, 0.51, 0.1, 0.52])
        result = Utilities.make_single_dimension(y_test)
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])
        self.assertEqual(0, result[2])
        self.assertEqual(1, result[3])

    def test_create_vector(self):
        x = np.array([[2, 5, 20, 15, 2], [3, 4, 19, 15]])
        result = Utilities.create_vector(x, 100)
        self.assertEqual(2, result[0][1])
        self.assertEqual(1, result[0][4])
        self.assertEqual(1, result[0][14])
        self.assertEqual(1, result[0][19])
        self.assertEqual(0, result[0][21])

        self.assertEqual(0, result[0][2])
        self.assertEqual(1, result[1][18])

    def test_measure_performance(self):
        y_test = np.array([1, 1, 1, 1, 1, 0, 0])
        y_actual = np.array([1, 1, 1, 1, 1, 1, 0])
        Utilities.measure_performance(y_test, y_actual)

    def test_measure_performance_auc(self):
        y_test = np.array([1,   1,   1,   1,   0, 0,    0, 0])
        y_actual = np.array([1,   0,   1,   1,   1,  1,    0, 0])
        y_actual_p = np.array([0.9, 0.49, 0.97, 0.7, 0.7, 0.55, 0, 0.45])
        vacc, vauc = Utilities.measure_performance_auc(y_test, y_actual, y_actual_p)
        self.assertEqual(0.62, round(vacc, 2))
        self.assertEqual(0.84, round(vauc, 2))

    def test_make_dual_convert(self):
        convertor = ClassConvertor("Test", {-2: 0, -1: 0, 0: 1, 1: 2, 2: 2})        
        self.assertEqual(0, convertor.is_supported(-2))
        self.assertEqual(0, convertor.is_supported(-1))
        self.assertEqual(1, convertor.is_supported(0))
        self.assertEqual(2, convertor.is_supported(1))
        self.assertEqual(2, convertor.is_supported(1))
