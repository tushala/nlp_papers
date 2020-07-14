# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.externals import joblib
from bayes_opt import BayesianOptimization
from gensim import models
from feature_pick import load_model
from const import *
max_length = 500  # 表示样本表示最大的长度,表示降维之后的维度
sentence_max_length = 1500  # 表示句子/样本在降维之前的维度
Train_features3, Test_features3, Train_label3, Test_label3 = [], [], [], []


tfidf_path, w2v_embedding, fast_embedding = load_model()
print("fast_embedding输出词表的个数{},w2v_embedding输出词表的个数{}".format(
    len(fast_embedding.wv.vocab.keys()), len(w2v_embedding.wv.vocab.keys())))

print("取词向量成功")

train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)
# print(train)
print("读取数据完成")
labelName = train.Label.unique()
print(labelName)


def Find_embedding_with_windows(embedding_matrix):
    '''
   函数说明：该函数用于获取在大小不同的滑动窗口(k=[2, 3, 4])， 然后进行平均或取最大操作。
   参数说明：
       - embedding_matrix：样本中所有词构成的词向量矩阵
   return: result_list 返回拼接而成的一维词向量
   '''
    result_list = []
    # ToDo:
    # 由于之前抽取的特征并没有考虑词与词之间交互对模型的影响，
    # 对于分类模型来说，贡献最大的不一定是整个句子， 可能是句子中的一部分， 如短语、词组等等。
    # 在此基础上我们使用大小不同的滑动窗口(k=[2, 3, 4])， 然后进行平均或取最大操作。
    return result_list


def Find_Label_embedding(word_matrix, label_embedding):
    '''
    函数说明：获取到所有类别的 label embedding， 与输入的 word embedding 矩阵相乘， 对其结果进行 softmax 运算，
            对 attention score 与输入的 word embedding 相乘的结果求平均或者取最大
            可以参考论文《Joint embedding of words and labels》获取标签空间的词嵌入
    parameters:
    -- example_matrix(np.array 2D): denotes the matrix of words embedding
    -- embedding(np.array 2D): denotes the embedding of all label in data
    return: (np.array 1D) the embedding by join label and word
    '''
    result_embedding=[]
    # To_Do
    # 第一步：基于consin相似度计算word embedding向量与label embedding之间的相似度
    # 第二步：通过softmax获取注意力分布
    # 第三步：将求得到的注意力分布与输入的word embedding相乘，并对结果进行最大化或求平均
    return result_embedding


def sentence2vec(query):
    '''
    函数说明：联合多种特征工程来构造新的样本表示，主要通过以下三种特征工程方法
            第一：利用word-embedding的average pooling和max-pooling
            第二：利用窗口size=2，3，4对word-embedding进行卷积操作，然后再进行max/avg-pooling操作
            第二：利用类别标签的表示，增加了词语和标签之间的语义交互，以此达到对词级别语义信息更深层次的考虑
            另外，对于词向量超过预定义的长度则进行截断，小于则进行填充
    参数说明：
    - query:数据集中的每一个样本
    return: 返回样本经过哦特征工程之后得到的词向量
    '''
    global max_length
    arr = []
    # 加载fast_embedding,w2v_embedding
    global fast_embedding, w2v_embedding
    fast_arr = np.array([fast_embedding.wv.get_vector(s)
                         for s in query if s in fast_embedding.wv.vocab.keys()])
    # 在fast_arr下滑动获取到的词向量
    if len(fast_arr) > 0:
        windows_fastarr = np.array(Find_embedding_with_windows(fast_arr))
        result_attention_embedding = Find_Label_embedding(
            fast_arr, fast_embedding)
    else:# 如果样本中的词都不在字典，则该词向量初始化为0
        # 这里300表示训练词嵌入设置的维度为300
        windows_fastarr = np.zeros(300)
        result_attention_embedding = np.zeros(300)

    fast_arr_max = np.max(np.array(fast_arr), axis=0) if len(
        fast_arr) > 0 else np.zeros(300)
    fast_arr_avg = np.mean(np.array(fast_arr), axis=0) if len(
        fast_arr) > 0 else np.zeros(300)

    fast_arr = np.hstack((fast_arr_avg, fast_arr_max))
    # 将多个embedding进行横向拼接
    arr = np.hstack((np.hstack((fast_arr, windows_fastarr)),
                     result_attention_embedding))
    global sentence_max_length
    # 如果样本的维度大于指定的长度则需要进行截取或者拼凑,
    result_arr = arr[:sentence_max_length] if len(arr) > sentence_max_length else np.hstack((
        arr, np.zeros(int(sentence_max_length-len(arr)))))
    return result_arr


query = train["Text"][0]
print(query)
fast_arr = np.array([fast_embedding.wv.get_vector(s)
                         for s in query if s in fast_embedding.wv.vocab.keys()])

print(fast_arr)