import numpy as np

from ..utilities import logger, Constants


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
            yield text, self.get_vector_from_tokens(tokens)
        else:
            for sentence in self.lexicon.review_to_sentences(text):
                yield sentence, self.get_vector_from_tokens(sentence)

    def get_vector_from_tokens(self, tokens):
        data = []
        for word in tokens:
            if word in self.word2vec.word_index:
                data.append(self.word2vec.word_index[word])
            elif Constants.UNK in self.word2vec.word_index:
                data.append(self.word2vec.word_index[Constants.UNK])

        if 0 in data:
            raise ValueError("Can't have zero index")
        data = np.array(data)
        if len(data) == 0:
            logger.debug("No Tokens Found")
            return None
        return data
