# -*- coding: utf-8 -*-
# skip-gram
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import re
from const import *
from huffman import HuffmanCoding

stoplist = stopwords.words('english')
flatten = lambda l: [item for sublist in l for item in sublist]
english_re = re.compile(r'[a-z]')


def get_corpus_words(n=800, k=2):
    corpus = list(nltk.corpus.gutenberg.sents(corpus_txt))[:n]
    corpus = [[word.lower() for word in sent if word.lower() not in stoplist and len(word) > 1] for sent in corpus if
              sent]
    corpus = [i for i in corpus if len(i) > 3]
    words = flatten(corpus)
    words_num = len(words)
    word_count = Counter(words)

    word_count = Counter(words).most_common(len(word_count) * 4 // 5)
    words = [(w, v) for (w, v) in word_count if v >= k and re.search(english_re, w)]

    return corpus, words, words_num


def get_index_dict(vocab):
    word2index = {'<UNK>': 0}

    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)
    index2word = {v: k for k, v in word2index.items()}
    return word2index, index2word


def prepare_word(words, word2index):
    return [word2index.get(word, word2index["<UNK>"]) for word in words]


def make_train_data(args):
    X_p = []
    y_p = []
    train_data = []
    corpus, words, words_count = get_corpus_words()
    vocab = [i[0] for i in words]
    vocab.append("<UNK>")
    # unk_num = words_count - sum(i[1] for i in words) # unk 太多了
    unk_num = words[0][1]
    words.append(("<UNK>", unk_num))
    words = dict(words)
    word2index, index2word = get_index_dict(vocab)
    windows = flatten(
        [list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in
         corpus])

    if args.model == "SG":  # skip-gram
        for window in windows:
            for i, w in enumerate(window):
                if i != WINDOW_SIZE:
                    train_data.append(([window[WINDOW_SIZE]], [window[i]]))
    elif args.model == "CB":  # CBOW
        for window in windows:
            train_data.append((window, [window[WINDOW_SIZE]]))
    else:
        train_data = get_copule_words_dict(windows, vocab, word2index)

    if args.model != "Glove":
        for tr in train_data:
            X_p.append(prepare_word(tr[0], word2index))
            y_p.append(prepare_word(tr[1], word2index))
        x = torch.tensor([i for i in X_p], dtype=torch.long)
        y = torch.tensor([i for i in y_p], dtype=torch.long)

        torch_dataset = TensorDataset(x, y)
    else:
        coocs = []
        weights = []
        for (xi, xj, cooc, weight) in train_data:
            X_p.append([xi])
            y_p.append([xj])
            coocs.append(cooc)
            weights.append(weight)
        x = torch.tensor([i for i in X_p], dtype=torch.long)
        y = torch.tensor([i for i in y_p], dtype=torch.long)
        c = torch.tensor([i for i in coocs], dtype=torch.float)
        w = torch.tensor([i for i in weights], dtype=torch.float)
        torch_dataset = TensorDataset(x, y, c, w)

    data_loader = DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    vocabs = prepare_word(vocab, word2index)
    res = (data_loader, vocabs, word2index, index2word)

    if args.us == "HS":  # Hierarchical Softmax
        hc = HuffmanCoding()

        hc.build(words)
        # todo
        tensor_2_dict = hc.get_d(word2index)
        tensor_2_dict = sorted(tensor_2_dict.items(), key=lambda x: len(x[1]))

        res = res + (tensor_2_dict,)

    elif args.us == "NS":  # 负采样
        ...
        # todo
    return res


def get_copule_words_dict(windows, vocab, word2index):
    # todo 统计全局
    from itertools import combinations_with_replacement
    pair_sta = defaultdict(int)
    for window in windows:
        center_words = window[WINDOW_SIZE]
        for i, w in enumerate(window):
            if i == WINDOW_SIZE:
                continue
            else:
                pair_sta[(center_words, w)] += 1
                pair_sta[(w, center_words)] += 1
    pair_wight_d = {}  # 无论是否存在都记录，未存在的weight很小
    for bigram in combinations_with_replacement(vocab, 2):
        weight = calc_weight(bigram[0], bigram[1], pair_sta)
        pair_wight_d[bigram] = weight
        pair_wight_d[(bigram[1], bigram[0])] = weight

    train_data = []
    for (xi, xj), weight_p in pair_wight_d.items():
        train_data.append([word2index[xi], word2index[xj], np.log(pair_sta[xi, xj] + 1), weight_p])

    return train_data


def calc_weight(w_i, w_j, d, x_max=100, alpha=0.75):
    # glove 中计算f(Xij)
    x_ij = d[(w_i, w_j)]
    x_ij = max(x_ij, 1)
    if x_ij < x_max:
        result = (x_ij / x_max) ** alpha
    else:
        result = 1
    return result


def calc_word_p(context_word_code, sig_vector):
    # sig_vector 已sigmoid后
    res = 1.
    for n, char in enumerate(context_word_code):
        if char == "1":
            res *= sig_vector[n]
        else:
            res *= (1 - sig_vector[n])
    return res
