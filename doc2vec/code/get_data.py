'''文字转换数字id'''
import os
import nltk
import numpy as np
import random
from collections import Counter
import pickle

WINDOW_SIZE = 10
PICKLE_PATH = "./data/pickle"


class Dataset:
    def __init__(self):
        if not os.path.exists(PICKLE_PATH + './test_labels.pkl'):
            train_datas, self.train_labels, train_unsup, test_datas, self.test_labels = self.get_all_datas()
            self.word2id = self.get_all_words(train_datas, train_unsup)
            self.train_datas = self.convert_data_word_to_id(self.word2id, train_datas)
            self.train_unsup = self.convert_data_word_to_id(self.word2id, train_unsup)
            self.test_datas = self.convert_data_word_to_id(self.word2id, test_datas)
            self.train_word_datas, self.train_para_datas, self.train_new_labels = self.convert_data_to_new_data(
                self.train_datas)
            self.test_word_datas, self.test_para_datas, self.test_new_labels = self.convert_data_to_new_data(
                self.test_datas)
            if not os.path.exists(PICKLE_PATH):
                os.mkdir(PICKLE_PATH)

            pickle.dump(self.word2id , open(PICKLE_PATH + "/word2id.pkl", "wb"))
            pickle.dump(self.train_word_datas, open(PICKLE_PATH + "/train_word_datas.pkl", "wb"))
            pickle.dump(self.train_para_datas, open(PICKLE_PATH + "/train_para_datas.pkl", "wb"))
            pickle.dump(self.train_new_labels, open(PICKLE_PATH + "/train_new_labels.pkl", "wb"))
            pickle.dump(self.train_labels, open(PICKLE_PATH + "/train_labels.pkl", "wb"))
            pickle.dump(self.test_word_datas, open(PICKLE_PATH + "/test_word_datas.pkl", "wb"))
            pickle.dump(self.test_para_datas, open(PICKLE_PATH + "/test_para_datas.pkl", "wb"))
            pickle.dump(self.test_new_labels, open(PICKLE_PATH + "/test_new_labels.pkl", "wb"))
            pickle.dump(self.test_labels, open(PICKLE_PATH + "/test_labels.pkl", "wb"))
        else:
            self.word2id = pickle.load(open(PICKLE_PATH + "/word2id.pkl", "rb"))
            self.train_word_datas = pickle.load(open(PICKLE_PATH + "/train_word_datas.pkl", "rb"))
            self.train_para_datas = pickle.load(open(PICKLE_PATH + "/train_para_datas.pkl", "rb"))
            self.train_para_datas = self.train_para_datas.reshape([self.train_para_datas.shape[0], 1])
            self.train_new_labels = pickle.load(open(PICKLE_PATH + "/train_new_labels.pkl", "rb"))
            self.train_labels = pickle.load(open(PICKLE_PATH + "/train_labels.pkl", "rb"))
            self.test_word_datas = pickle.load(open(PICKLE_PATH + "/test_word_datas.pkl", "rb"))
            self.test_para_datas = pickle.load(open(PICKLE_PATH + "/test_para_datas.pkl", "rb"))
            self.test_para_datas = self.test_para_datas.reshape([self.test_para_datas.shape[0], 1])
            self.test_new_labels = pickle.load(open(PICKLE_PATH + "/test_new_labels.pkl", "rb"))
            self.test_labels = pickle.load(open(PICKLE_PATH + "/test_labels.pkl", "rb"))
    def get_data(self, path):
        datas = []
        paths = os.listdir(path)
        paths = [path + file_name for file_name in paths]
        for i, file in enumerate(paths):
            data = open(file, 'r', encoding='utf-8').read()
            data = data.lower()
            data = nltk.word_tokenize(data)
            datas.append(data)
        return datas
    
    def get_all_datas(self):
        '''
        得到所有的训练句子，无监督句子和测试句子。
        :return: 返回训练句子，训练标签，无监督句子，测试句子，测试标签
        '''
        train_neg_datas = self.get_data('data/train/neg/')
        train_pos_datas = self.get_data("data/train/pos/")
        train_unsup = self.get_data("data/train/unsup/")
        test_neg_datas = self.get_data("data/test/neg/")
        test_pos_datas = self.get_data("data/test/pos/")
        train_datas = train_neg_datas + train_pos_datas
        train_labels = [0] * len(train_neg_datas) + [1] * len(train_pos_datas)
        test_datas = test_neg_datas + train_pos_datas
        test_labels = [0] * len(test_neg_datas) + [1] * len(test_pos_datas)
        tmp = list(zip(train_datas, train_labels))
        random.shuffle(tmp)
        train_datas, train_labels = zip(*tmp)
        tmp = list(zip(test_datas, test_labels))
        random.shuffle(tmp)
        test_datas, test_labels = zip(*tmp)
        # print(len(train_datas), len(train_labels))
        # print(len(train_unsup))
        # print(len(test_datas), len(test_labels))
        return train_datas, train_labels, train_unsup, test_datas, test_labels
    
    def convert_data_word_to_id(self, word2id, datas):
        # 将 datas 里的词都转化正对应的id
        for i, sentence in enumerate(datas):
            for j, word in enumerate(sentence):
                datas[i][j] = word2id.get(word, 1)
        return datas
    
    def get_all_words(self, train_datas, train_unsup):
        all_words = []
        for sentence in train_datas:
            all_words.extend(sentence)
        for sentence in train_unsup:
            all_words.extend(sentence)
        count = Counter(all_words)
        count = dict(count.most_common(29998))
        word2id = {"<pad>": 0, "<unk>": 1}
        for word in count:
            word2id[word] = len(word2id)
        return word2id
    
    def convert_data_to_new_data(self, datas):
        '''
        获取句子id， 句子label
        '''
        new_word_datas = []
        new_papr_datas = []
        new_labels = []
        for i, data in enumerate(datas):
            data_length = len(data)
            if data_length < WINDOW_SIZE:
                tmp_words = [0] * (WINDOW_SIZE - data_length) + data[:-1]
                if set(tmp_words) == {1}:  # 舍去连续9个词都是unk
                    break
                new_word_datas.append(tmp_words)
                new_papr_datas.append(i)
                new_labels.append(data[-1])
                continue
            for j in range(data_length):
                tmp_words = data[j: j + WINDOW_SIZE - 1]
                if set(tmp_words) == {1}:
                    continue
                new_papr_datas.append(i)
                new_word_datas.append(tmp_words)
                new_labels.append(data[j + 9])
                if j == data_length - WINDOW_SIZE:
                    break
        new_word_datas = np.array(new_word_datas)
        new_papr_datas = np.array(new_papr_datas)
        new_labels = np.array(new_labels)
        print('new_word_datas.shape: ', new_word_datas.shape)
        print('new_papr_datas.shape: ', new_papr_datas.shape)
        print('new_labels.shape: ', new_labels.shape)
        return new_word_datas, new_papr_datas, new_labels


if __name__ == '__main__':
    d = Dataset()
