from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Activation, Input
from keras.layers import LSTM, Bidirectional, Embedding, concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop


from keras.utils import plot_model

class TextGAN:
    def __init__(self, vocabulary, max_seq_len=20, embedding_size=256, embedding_matrix=None):
        self.vocabulary = vocabulary

        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix

    def residual_block(self, x1, i):
        x = Bidirectional(LSTM(int(self.embedding_size/2), return_sequences=True, kernel_initializer='glorot_uniform'), name='residual_block_{}_1'.format(i))(x1)
        x = Bidirectional(LSTM(int(self.embedding_size/2), return_sequences=True, kernel_initializer='glorot_uniform'), name='residual_block_{}_2'.format(i))(x)

        return concatenate([x1, x])

    def build_generator(self):
        inp = Input(shape=(self.max_seq_len, self.vocabulary,), name='sentence_input')

        embed = Dense(int(self.embedding_size), kernel_initializer='glorot_uniform', activation='sigmoid', name='embedding_layer')(inp)
        x = concatenate([embed, embed], axis=-1)

        #if self.embedding_matrix != None:
        #    x = Embedding(self.vocabulary, self.embedding_size, mask_zero=False, weights=[self.embedding_matrix])(inp)
        #else:
        #    x = Embedding(self.vocabulary, self.embedding_size, mask_zero=False)(inp)

        x = Bidirectional(LSTM(int(self.embedding_size/2), return_sequences=True, kernel_initializer='glorot_uniform'))(x)
        x = self.residual_block(x, 1)
        x = self.residual_block(x, 2)

        x = Bidirectional(LSTM(int(self.embedding_size/2), return_sequences=True, kernel_initializer='glorot_uniform'))(x)

        x = Dense(self.vocabulary, kernel_initializer='glorot_uniform', activation='softmax')(x)

        return Model(inputs=inp, outputs=x)

    def build_discriminator(self):
        actual_sentence_input = Input(shape=(self.max_seq_len, self.vocabulary), name='sentence_input')
        paraphrase_input = Input(shape=(self.max_seq_len, self.vocabulary), name='paraphrase_input')

        # if self.embedding_matrix != None:
        #     embedding_layer = Embedding(self.vocabulary, self.embedding_size, mask_zero=False, weights=[self.embedding_matrix])
        # else:
        #     embedding_layer = Embedding(self.vocabulary, self.embedding_size, mask_zero=False)

        #actual_sentence = embedding_layer(actual_sentence_input)
        #paraphrase = embedding_layer(paraphrase_input)

        embedding_layer = Dense(self.embedding_size, activation='sigmoid', use_bias=False, name='embedding_layer')

        x = concatenate([embedding_layer(actual_sentence_input), embedding_layer(paraphrase_input)], axis=-1)


        x1 = Bidirectional(LSTM(int(self.embedding_size/2), return_sequences=True, kernel_initializer='glorot_uniform'))(actual_sentence_input)
        x2 = Bidirectional(LSTM(int(self.embedding_size/2), return_sequences=True, kernel_initializer='glorot_uniform'))(paraphrase_input)

        x = concatenate([x1, x2])

        x = Bidirectional(LSTM(int(self.embedding_size/2), return_sequences=True, kernel_initializer='glorot_uniform'))(x)
        x = Bidirectional(LSTM(int(self.embedding_size/2), return_sequences=False, kernel_initializer='glorot_uniform'))(x)

        x = Dense(2, kernel_initializer='glorot_uniform', activation='softmax')(x)

        return Model(inputs=[actual_sentence_input, paraphrase_input], outputs=x)

    def build_gan(self):
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        plot_model(self.discriminator, to_file='discriminator.png')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


        self.generator = self.build_generator()
        plot_model(self.generator, to_file='generator.png')
        actual_sentence = Input(shape=(self.max_seq_len, self.vocabulary), name='sentence_input_generator')
        paraphrase = self.generator(actual_sentence)

        #paraphrase = output[0]

        self.discriminator.trainable = False

        valid = self.discriminator([actual_sentence, paraphrase])

        self.text_gan = Model(inputs=actual_sentence, outputs=valid)
        plot_model(self.text_gan, to_file='combined.png')
        self.text_gan.compile(loss='binary_crossentropy', optimizer=optimizer)


if __name__ == '__main__':
    gan = TextGAN(65000)
    generator = gan.build_generator()
    generator.summary()

    print()
    print()

    discriminator = gan.build_discriminator()
    discriminator.summary()

    print()
    print()

    gan.build_gan()
    gan.text_gan.summary()
