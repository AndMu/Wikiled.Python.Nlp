import unittest

from wikilednlp.Discovery import seeds


class SentimentTableTests(unittest.TestCase):

    def test_twitter_seeds(self):
        result = seeds.twitter_seeds()
        self.assertEqual(31, len(result[0]))
        self.assertEqual(33, len(result[1]))

    def test_construct(self):
        table = seeds.construct(seeds.twitter_seeds)
        self.assertEqual(31, len(table.positive))
        self.assertEqual(33, len(table.negative))