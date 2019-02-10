import abc
import gc
from abc import ABC, abstractmethod
import os
from os import path, makedirs
from keras.initializers import Constant
from keras import backend as k, Input, Model
from keras import callbacks
from keras.layers import Embedding, Dropout, Dense, np, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, \
    Bidirectional
from keras_preprocessing import sequence
from pathlib2 import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

import wikilednlp.utilities.Constants as Constants
from wikilednlp.learning import logger
from wikilednlp.utilities.Utilities import Utilities

seed = 7
np.random.seed(seed)


class BaseDeepStrategy(ABC):
    __metaclass__ = abc.ABCMeta

    counter = 0

    def __init__(self, project_name, loader, sub_project, max_length):
        self.counter = BaseDeepStrategy.counter
        BaseDeepStrategy.counter += 1
        self.max_length = max_length
        self.project_path = path.join(Constants.TEMP, 'Deep', project_name, sub_project)        
        logger.info('%s with doc_size %i to [%s]', project_name, max_length, self.project_path)        
        self.epochs_number = 20
        self.model = None
        self.early_stop = None
        self.output_drop_out = 0.5
        self.loader = loader
        self.total_classes = self.loader.convertor.total_classes()

    def __populate__(self, copy_instance):
        copy_instance.early_stop = self.early_stop

    def get_embeddings(self):

        logger.info('get_embeddings')
        vectors = self.loader.parser.word2vec.embedding_matrix
        embedding_layer = Embedding(vectors.shape[0],
                                    vectors.shape[1],
                                    embeddings_initializer=Constant(vectors),
                                    input_length=self.max_length,
                                    trainable=False)
        return embedding_layer

    def add_output(self, model):
        if self.output_drop_out is not None:
            model = Dropout(self.output_drop_out)(model)
        output = Dense(self.total_classes, activation='softmax', name='Main_Output')(model)
        return output

    def init_mode(self):
        if not hasattr(self, 'model') or self.model is None:
            self.model = self.construct_model()
            self.model.summary()
            self.compile()

    def get_file_name(self, name):
        file_path = path.join(self.project_path, 'weights', name, str(self.max_length) + '_keras-lstm.h5')
        weights = path.dirname(file_path)
        if not Path(weights).exists():
            makedirs(weights)
        return file_path

    def load(self, name):
        self.init_mode()
        cache_file_name = self.get_file_name(name)
        if path.exists(cache_file_name):
            logger.info('Loading weights [%s]...', cache_file_name)
            self.model.load_weights(cache_file_name)
        else:
            logger.info('Weights file not found - [%s]...', cache_file_name)

    def save(self, name):
        cache_file_name = self.get_file_name(name)
        logger.info('Saving weights [%s]...', cache_file_name)
        self.model.save_weights(cache_file_name)

    def delete_weights(self, name):
        cache_file_name = self.get_file_name(name)
        if Path(cache_file_name).exists():
            logger.info('Deleting weight %s', cache_file_name)
            os.remove(cache_file_name)

    def test(self, test_x, test_y):
        logger.info('Testing with %i records', len(test_x))
        self.init_mode()
        test_x = sequence.pad_sequences(test_x, maxlen=self.max_length)
        loss, acc = self.model.evaluate(test_x, test_y, Constants.TESTING_BATCH)
        logger.info('Test loss / Test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    def predict_proba(self, test_x):
        logger.info('Predict with %i records', len(test_x))
        self.init_mode()
        test_x = sequence.pad_sequences(test_x, maxlen=self.max_length)
        y = self.model.predict(test_x, batch_size=Constants.TESTING_BATCH, verbose=1)
        return y

    def predict(self, test_x):
        logger.info('Predict with %i records', len(test_x))
        y_prob = self.predict_proba(test_x)
        return y_prob.argmax(axis=-1)

    @abstractmethod
    def construct_model(self):
        pass

    def test_predict(self, test_x, test_y):
        logger.info('Test predict_proba with %i records', len(test_x))
        result_y_prob = self.predict_proba(test_x)
        result_y = Utilities.make_single_dimension(result_y_prob)
        result_y_prob_single = self.loader.convertor.make_single(result_y_prob)

        Utilities.measure_performance(test_y, result_y)
        Utilities.measure_performance_auc(test_y, result_y, result_y_prob)

        return result_y, result_y_prob_single

    def get_classes(self, y):
        single = Utilities.make_single_dimension(y)
        return np.unique(check_array(single, ensure_2d=False, allow_nd=True))

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    def fit(self, train_x, train_y):
        logger.info('Training with %i records', len(train_x))
        self.init_mode()
        Utilities.count_occurences(train_y)
        train_x, train_y = Utilities.unison_shuffled_copies(train_x, train_y)
        train_x = sequence.pad_sequences(train_x, maxlen=self.max_length)
        if len(train_y.shape) == 1:
            train_y = Utilities.make_dual(train_y, self.total_classes)
        cbks = None
        if self.early_stop is not None:
            logger.info('Will use early stop with: %i', self.early_stop)
            cbks = [callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stop)]
        self.model.fit(train_x, train_y, batch_size=Constants.TRAINING_BATCH, callbacks=cbks, epochs=self.epochs_number,
                   validation_split=0.25, shuffle=True)


class CnnSentiment(BaseDeepStrategy):

    def __init__(self, loader, project_name, max_length, vocab_size=10000):
        self.max_length = max_length
        self.vocab_size = vocab_size        
        self.loader = loader
        self.word_vector_size = loader.parser.word2vec.vector_size
        super(CnnSentiment, self).__init__(project_name, loader, self.get_name(), max_length)

    def get_name(self):
        return self.loader.parser.word2vec.name + '_CnnSentiment'

    def construct_model(self):

        k.set_image_data_format('channels_first')

        sequence_input = Input(shape=(self.max_length,))
        embedded_sequences = self.get_embeddings()(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)

        preds = self.add_output(x)
        model = Model(sequence_input, preds)

        # Inner Product layer (as in regular neural network, but without non-linear activation function)
        return model

    def copy(self):
        copy_instance = CnnSentiment(self.loader, self.project_name, self.max_length, self.vocab_size)
        self.__populate__(copy_instance)
        return copy_instance

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()


class LSTMSentiment(CnnSentiment):

    def __init__(self, loader, project_name, max_length, vocab_size=10000, lstm_size=100):
        self.lstm_size = lstm_size
        super(LSTMSentiment, self).__init__(loader, project_name, max_length, vocab_size)

    def get_name(self):
        return '{}_LSTM_{}_{}'.format(self.loader.parser.word2vec.name, self.max_length, self.lstm_size)

    def construct_model(self):
        sequence_input = Input(shape=(self.max_length,))
        embedded_sequences = self.get_embeddings()(sequence_input)
        x = LSTM(self.lstm_size, return_sequences=False)(embedded_sequences)
        preds = self.add_output(x)
        model = Model(sequence_input, preds)
        return model

    def copy(self):
        copy_instance = LSTMSentiment(self.loader, self.project_name, self.max_length, self.vocab_size, self.lstm_size)
        self.__populate__(copy_instance)
        return copy_instance

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()


class BidiLTSMSentiment(LSTMSentiment):

    def get_name(self):
        return '{}_ENCODER_{}_{}'.format(self.loader.parser.word2vec.name, self.max_length, self.lstm_size)

    def construct_model(self):
        sequence_input = Input(shape=(self.max_length,))
        embedded_sequences = self.get_embeddings()(sequence_input)
        x = Bidirectional(LSTM(self.lstm_size))(embedded_sequences)
        preds = self.add_output(x)
        model = Model(sequence_input, preds)
        return model

    def copy(self):
        copy_instance = BidiLTSMSentiment(self.loader, self.project_name, self.max_length, self.vocab_size, self.lstm_size)
        self.__populate__(copy_instance)        
        return copy_instance

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()


class DeepSklearnWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, x, y):
        y = self.classifier.loader.convertor.create_vector(y)
        result = self.classifier.fit(x, y)
        self.classifier.save('Multi_' + str(self.classifier.counter))
        self.classes_ = self.classifier.get_classes(y)
        del self.classifier.model
        gc.collect()
        return result

    def predict_proba(self, x):
        self.classifier.load("Multi_" + str(self.classifier.counter))
        y = self.classifier.predict_proba(x)
        del self.classifier.model
        gc.collect()
        return y