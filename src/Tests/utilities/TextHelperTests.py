import unittest

from ddt import data, unpack, ddt

from wikilednlp.utilities.TextHelper import TextHelper


@ddt
class TextHelperTests(unittest.TestCase):

    @data(['emoticon_cool', True], ['cool', False])
    @unpack
    def is_emoticon_test(self, text, result):
        is_emoticon = TextHelper.is_emoticon(text)
        self.assertEqual(result, is_emoticon)

    @data(['#cool', True], ['cool', False], ['#', False])
    @unpack
    def is_hash_test(self, text, result):
        is_hash = TextHelper.is_hash(text)
        self.assertEqual(result, is_hash)
