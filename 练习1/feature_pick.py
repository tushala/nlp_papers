# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
from const import *
from gensim import models
from sklearn.externals import joblib


def trainer_tfidf(path):
    corpus = get_tf_idf_corpus(path)
    count_vect = TfidfVectorizer()
    # 第二步：用模型对象去fit训练数据集

    tfidf_model = count_vect.fit(corpus)
    sparse_result = tfidf_model.transform(corpus)
    print('train tfidf_embedding')
    # 返回是一个稀疏矩阵
    return sparse_result
    # return tfidf.toarray()


# w = trainer_tfidf(train_data_path)
# print(w)
# print(w.data.shape)

def trainer_w2v():
    w2v = models.Word2Vec(min_count=2,
                          window=3,
                          size=300,
                          sample=6e-5,
                          alpha=0.03,
                          min_alpha=0.0007,
                          negative=15,
                          workers=4,
                          iter=10,
                          max_vocab_size=50000)

