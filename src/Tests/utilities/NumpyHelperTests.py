import unittest
import numpy as np
from wikilednlp.utilities.NumpyHelper import NumpyDynamic


class NumpyDynamicTests(unittest.TestCase):

    def empty_test(self):
        data = NumpyDynamic(np.int32)
        result = data.finalize()
        self.assertEqual(0, len(result))

    def add_test(self):
        data = NumpyDynamic(np.int32, (100, 3))
        for i in range(0, 110):
            data.add(np.array([1, 2, 3]))
        result = data.finalize()
        self.assertEqual(110, len(result))

    def add_test_multi(self):
        data = NumpyDynamic(np.int32, (100, 2, 3))
        for i in range(0, 110):
            data.add(np.array([[1, 2, 3], [1, 2, 3]]))
        result = data.finalize()
        self.assertEqual(110, len(result))
