from keras import Input, Model
from keras.layers import LSTM, Dense
import numpy as np
from wikilednlp.learning.DeepLearning import CnnSentiment
from wikilednlp.utilities import Constants
from wikilednlp.learning import logger


class Seq2Seq(CnnSentiment):
    def get_name(self):
        return self.loader.parser.word2vec.name + '_SEQ2SEQ_' + str(self.max_length)

    def construct_model(self):

        vocab_size = self.get_vocab_size() + 1
        self.encoder_inputs = Input(shape=(None,))
        en_x = self.get_embeddings()(self.encoder_inputs)
        encoder = LSTM(50, return_state=True)
        encoder_outputs, state_h, state_c = encoder(en_x)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        dex = self.get_embeddings()
        final_dex = dex(decoder_inputs)
        decoder_lstm = LSTM(50, return_sequences=True, return_state=True)

        decoder_outputs, _, _ = decoder_lstm(final_dex, initial_state=self.encoder_states)

        decoder_dense = Dense(vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([self.encoder_inputs, decoder_inputs], decoder_outputs)
        return model


    def get_decoder(self):
        # DECODER STATES

        vocab_size = self.get_vocab_size() + 1

        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)
        self.encoder_model.summary()

        decoder_state_input_h = Input(shape=(50,))
        decoder_state_input_c = Input(shape=(50,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_inputs = Input(shape=(None,))
        dex = self.get_embeddings()
        final_dex2 = dex(decoder_inputs)
        decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
        decoder_dense = Dense(vocab_size, activation='softmax')

        decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

    def compile(self):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def fit(self, input_texts, target_texts):

        self.init_mode()

        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) + 2 for txt in target_texts])

        if max_decoder_seq_length > self.max_length + 2:
            logger.info("Target text length is too large - decreasing")
            max_decoder_seq_length = self.max_length + 2

        if max_encoder_seq_length > self.max_length:
            logger.info("Source text length is too large - decreasing")
            max_encoder_seq_length = self.max_length

        vocab_size = self.get_vocab_size()
        encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype='float32')
        decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length), dtype='float32')
        decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, vocab_size + 1), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, word in enumerate(input_text[0:self.max_length]):
                encoder_input_data[i, t] = word
            target = np.concatenate([[Constants.START_ID], target_text[0:self.max_length], [Constants.END_ID]])
            for t, word in enumerate(target):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t] = word
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, word] = 1.
        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                       batch_size=Constants.TRAINING_BATCH, epochs=self.epochs_number, validation_split=0.2)

    def decode_sequence(self, input_seq):

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = Constants.START_ID

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            if sampled_token_index == Constants.END_ID or len(decoded_sentence) > self.max_length:
                break

            sampled_char = self.loader.parser.word2vec.index_word[sampled_token_index]
            decoded_sentence += ' ' + sampled_char

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return decoded_sentence

