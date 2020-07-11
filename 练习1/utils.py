# -*- coding: utf-8 -*-
import jieba
from const import *
import csv
from data_proc import stopwords
import random


def create_data(path):
    data = csv.reader(open(path, encoding="utf-8"))
    data_list = []
    for n, d in enumerate(data):
        if n == 0:
            continue
        _, label, text = d
        text = han_compile.findall(text)
        text = "".join(text)
        text = jieba.cut(text)
        text = [i for i in text if i not in stopwords]
        data_list.append((Labels2Level[label], " ".join(text)))
    random.shuffle(data_list)
    train_length = int(len(data_list) * 0.8)
    dev_length = int(len(data_list) * 0.1)
    test_length = int(len(data_list) * 0.1)
    train_data = data_list[:train_length]
    dev_data = data_list[train_length + 1:train_length + 1 + dev_length]
    test_data = data_list[train_length + 1 + dev_length:]
    headers = ['Label', 'Text']
    for s in save_list:
        data = locals()[f'{s}_data']
        with open(os.path.join(data_path, f"{s}.csv"), 'w', encoding="utf-8", newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(data)


def get_tf_idf_corpus(path):
    data = csv.reader(open(path, encoding="utf-8"))
    data_list = []
    for n, d in enumerate(data):
        if n == 0:
            continue
        _, text = d
        data_list.append(text)
    return data_list
