import gensim

from src.wikilednlp.embeddings.VectorManagers import WordVecManager
from src.wikilednlp.utilities import Constants

if __name__ == '__main__':
    Constants.set_special_tags(False)
    path = 'e:/DataSets/word2vec'
    model = gensim.models.Word2Vec.load(f'{path}/Imdb_min2.bin')
    model.wv.save_word2vec_format(f'{path}/Imdb_min2_new.bin', f'{path}/Imdb_min2_new.dic', binary=True)
    manager = WordVecManager(f'{path}/Imdb_min2.bin', vocab_size=30000)
    manager.save_index(f'{path}/Imdb_min2_new.idx')

