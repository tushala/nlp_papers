# -*- coding: utf-8 -*-
# skip-gram
import nltk
from nltk.corpus import stopwords
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import torch
import re
from const import *

stoplist = stopwords.words('english')
flatten = lambda l: [item for sublist in l for item in sublist]
english_re = re.compile(r'[a-z]')


def get_corpus_words(n=300, k=1):
    corpus = list(nltk.corpus.gutenberg.sents(corpus_txt))[:n]
    corpus = [[word.lower() for word in sent if word.lower() not in stoplist and len(word) > 1] for sent in corpus if
              sent]
    corpus = [i for i in corpus if len(i) > 3]
    words = flatten(corpus)
    word_count = Counter(words)

    word_count = Counter(words).most_common(len(word_count) * 4 // 5)
    words = [w for (w, v) in word_count if v >= k and re.search(english_re, w)]
    words.append("<UNK>")
    return corpus, words


def get_index_dict(vocab):
    word2index = {'<UNK>': 0}

    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)
    index2word = {v: k for k, v in word2index.items()}
    return word2index, index2word


def prepare_word(words, word2index):

    return [word2index.get(word, word2index["<UNK>"]) for word in words]


def make_train_data(sg=True):
    X_p = []
    y_p = []
    train_data = []
    corpus, vocab = get_corpus_words()

    word2index, index2word = get_index_dict(vocab)
    windows = flatten(
        [list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in
         corpus])
    if sg:  # skip-gram
        for window in windows:
            for i, w in enumerate(window):
                if i != WINDOW_SIZE:
                    train_data.append(([window[WINDOW_SIZE]], [window[i]]))

    else:  # CBOW
        for window in windows:
            train_data.append((window, [window[WINDOW_SIZE]]))

    for tr in train_data:
        X_p.append(prepare_word(tr[0], word2index))
        y_p.append(prepare_word(tr[1], word2index))
    x = torch.tensor([i for i in X_p], dtype=torch.long)
    y = torch.tensor([i for i in y_p], dtype=torch.long)

    torch_dataset = TensorDataset(x, y)
    data_loader = DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    vocabs = prepare_word(vocab, word2index)
    return data_loader, vocabs, word2index, index2word
