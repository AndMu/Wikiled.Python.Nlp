import numpy as np

from wikilednlp.utilities import logger, Constants


class EmbeddingVecSource(object):

    def __init__(self, lexicon, word2vec, use_sentence=False):
        self.lexicon = lexicon
        self.word2vec = word2vec
        self.use_sentence = use_sentence

    def get_vector(self, file_name):
        with open(file_name, encoding='utf-8') as myfile:
            text = myfile.read()
            text = text.replace('\n', '')
            return self.get_vector_from_review(text)

    def get_vector_from_review(self, text):
        if not self.use_sentence:
            tokens = self.lexicon.review_to_wordlist(text)
            yield self.get_vector_from_tokens(tokens)
        else:
            for sentence in self.lexicon.review_to_sentences(text):
                yield self.get_vector_from_tokens(sentence)

    def get_vector_from_tokens(self, tokens):
        data = list(self.word2vec.word_index[word] for word in tokens if word in self.word2vec.word_index)
        if Constants.EMBEDDING_START_INDEX == 1 and 0 in data:
            raise ValueError("Can't have zero")
        data = np.array(data)
        if len(data) == 0:
            logger.debug("No Tokens Found")
            return None
        return data
