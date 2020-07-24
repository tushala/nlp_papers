# -*- coding: utf-8 -*-
from src.utils import *
from src.mylib.const import *

import re
from torchtext import data, vocab
import jieba
import logging

jieba.setLogLevel(logging.INFO)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


def word_cut(text):
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev = data.TabularDataset.splits(
        path=path, format='csv', skip_header=True,
        train='train.csv', validation='dev.csv',
        fields=[
            ('index', None),
            ('text', text_field),
            ('label', label_field)
        ]
    )
    return train, dev

#
# text_field = data.Field(lower=True)
# label_field = data.Field(sequential=False)
# train_dataset, dev_dataset = get_dataset(data_path, text_field, label_field)
# print(train_dataset.fields.text.__dict__)
