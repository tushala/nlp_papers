# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
from const import *
from gensim import models
from sklearn.externals import joblib


def trainer_tfidf(path):
    corpus = get_corpus(path, tf_idf=True)
    count_vect = TfidfVectorizer()
    # 第二步：用模型对象去fit训练数据集

    tfidf_model = count_vect.fit(corpus)
    sparse_result = tfidf_model.transform(corpus)
    print('train tfidf_embedding')
    # 返回是一个稀疏矩阵
    return sparse_result
    # return tfidf.toarray()


def trainer_w2v(path):
    corpus = get_corpus(path, w2v=True)
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
    w2v.build_vocab(corpus)
    w2v.train(corpus,
              total_examples=w2v.corpus_count,
              epochs=15,
              report_delay=1)

    print('train fast_embedding')
    return w2v


def trainer_fasttext(path):
    corpus = get_corpus(path, w2v=True)
    fast = models.FastText(corpus, size=300, window=3, min_count=2)
    return fast


def saver(path):
    '''
    函数说明：该函数存储训练好的模型
    '''
    # hint: 通过joblib.dump保存tfidf
    tf_idf = trainer_tfidf(path)

    joblib.dump(tf_idf, tfidf_path)
    print('save tfidf model')
    # hint: w2v可以通过自带的save函数进行保存
    w2v = trainer_w2v(path)
    joblib.dump(w2v, w2v_path)
    print('save word2vec model')
    # hint: fast可以通过自带的save函数进行保存
    fast = trainer_fasttext(path)
    joblib.dump(fast, fasttext_path)
    print('save fast model')


def load_model():
    '''
    函数说明：该函数加载训练好的模型
    '''
    # ToDo
    # 加载模型
    # hint: tfidf可以通过joblib.load进行加载
    # w2v和fast可以通过gensim.models.KeyedVectors.load加载
    print('load tfidf_embedding model')
    tfidf = joblib.load(tfidf_path)
    print('load w2v_embedding model')
    w2v = joblib.load(w2v_path)
    print('load fast_embedding model')
    fast = joblib.load(fasttext_path)
    return tfidf, w2v, fast


# tfidf_path, w2v_path, fast = load_model()