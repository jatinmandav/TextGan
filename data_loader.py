import random
import numpy as np
from gensim.models import FastText
import pandas as pd
import os

from tqdm import tqdm

from keras.preprocessing.text import Tokenizer

class Preprocess:
    def __init__(self, raw_text, max_seq_len=20):
        self.max_seq_len = max_seq_len
        text = []

        with open(raw_text, 'r') as f:
            text += f.read().split('\n')

        self.tokenizer = Tokenizer(text)
        self.tokenizer.fit_on_texts(text)

        self.word_to_index = self.tokenizer.word_index
        self.word_to_index['_PAD_'] = len(self.word_to_index) + 1
        self.index_to_word = dict([(i, word) for word, i in self.word_to_index.items()])

        self.vocab_size = len(self.word_to_index)

        print('Vocabulary Size: {}'.format(self.vocab_size))

    def process(self, file, mode='train'):
        df = pd.read_csv(file, sep='\t').head(500)
        if mode == 'train':
            self.train_data = []
        else:
            self.test_data = []

        for i in tqdm(range(len(df))):
            data = []
            for j in range(0, 6):
                if isinstance(df[str(j)][i], str):
                    data.append(df[str(j)][i])

            if mode == 'train':
                self.train_data.append(data)
            else:
                self.test_data.append(data)

        print('{} processed.'.format(file))

    def positive_batch(self, mode='train', batch_size=32):
        sentence_batch = []
        paraphrase_batch = []
        for _ in range(batch_size):
            if mode == 'train':
                sentence_pair = []
                while len(sentence_pair) < 2:
                    index = random.randint(0, len(self.train_data) - 1)
                    sentence_pair = self.train_data[index]
            else:
                sentence_pair = []
                while len(sentence_pair) < 2:
                    index = random.randint(0, len(self.test_data) - 1)
                    sentence_pair = self.test_data[index]

            sentence = random.choice(sentence_pair)
            sentence_pair.remove(sentence)
            sentence = sentence.split(' ')
            paraphrase = random.choice(sentence_pair).split(' ')

            sent = []
            for i, word in enumerate(sentence):
                if i < self.max_seq_len:
                    try:
                        one_hot = np.zeros(self.vocab_size)
                        one_hot[self.word_to_index[word]-1] = 1.
                        sent.append(one_hot)
                    except KeyError:
                        pass

            padding = self.max_seq_len - len(sent)
            for i in range(padding):
                try:
                    one_hot = np.zeros(self.vocab_size)
                    one_hot[self.word_to_index['_PAD_']-1] = 1.
                    sent.append(one_hot)
                except KeyError:
                    pass
                #sent.append(self.word_to_index['_PAD_']-1)

            sentence_batch.append(sent)

            sent = []
            for i, word in enumerate(paraphrase):
                if i < self.max_seq_len:
                    try:
                        one_hot = np.zeros(self.vocab_size)
                        one_hot[self.word_to_index[word]-1] = 1.
                        sent.append(one_hot)
                    except KeyError:
                        pass

            padding = self.max_seq_len - len(sent)
            for i in range(padding):
                try:
                    one_hot = np.zeros(self.vocab_size)
                    one_hot[self.word_to_index['_PAD_']-1] = 1.
                    sent.append(one_hot)
                except KeyError:
                    pass
                #sent.append(self.word_to_index['_PAD_'])

            paraphrase_batch.append(sent)

        return np.array(sentence_batch), np.array(paraphrase_batch)

    def load_embedding(self, fasttext_bin, embedding_size=256):
        self.embedding = FastText.load(fasttext_bin)
        self.vocab_size = len(self.embedding.wv.vocab) + 1
        self.embedding_size = embedding_size

        # self.embedding_matrix = np.zeros((self.vocab_size, embedding_size))
        #
        # for word, i in self.word_to_index.items():
        #     try:
        #         if word == '_PAD_':
        #             self.embedding_matrix[i] = np.zeros(embedding_size)
        #         else:
        #             embedding_vector = embedding[word]
        #             if embedding_vector is not None:
        #                 self.embedding_matrix[i] = embedding_vector
        #     except Exception as e:
        #         pass
        #
        # print('Embedding Matrix Loaded. Size: {}'.format(self.embedding_matrix.shape))


if __name__ == '__main__':
    processor = Preprocess('raw_text.txt', max_seq_len=8)
    processor.process('test_data.tsv')

    sent, para = processor.positive_batch(batch_size=20)
    print(sent.shape, para.shape)
