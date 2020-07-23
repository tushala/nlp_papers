# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
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

tfidf_model, w2v_embedding, fast_embedding = load_model()  # todo tf_idf 不会

print("fast_embedding输出词表的个数{},w2v_embedding输出词表的个数{}".format(
    len(fast_embedding.wv.vocab.keys()), len(w2v_embedding.wv.vocab.keys())))

print("取词向量成功")

train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)
# print(train)
print("读取数据完成")
labelName = train.Label.unique()
print(labelName)


def Find_embedding_with_windows(embedding_matrix, mean=True, k=(2, 3, 4)):
    '''
   函数说明：该函数用于获取在大小不同的滑动窗口(k=[2, 3, 4])， 然后进行平均或取最大操作。
   参数说明：
       - embedding_matrix：样本中所有词构成的词向量矩阵
   return: result_list 返回拼接而成的一维词向量
   '''
    length = len(embedding_matrix)
    if length == 1:
        result_list = embedding_matrix
    else:
        result_list = []
        for _k in k:
            for i in range(length - _k + 1):
                cur_embedding = np.mean(embedding_matrix[i:i + _k], axis=0)
                result_list.append(cur_embedding)
    # ToDo:
    # 由于之前抽取的特征并没有考虑词与词之间交互对模型的影响，
    # 对于分类模型来说，贡献最大的不一定是整个句子， 可能是句子中的一部分， 如短语、词组等等。
    # 在此基础上我们使用大小不同的滑动窗口(k=[2, 3, 4])， 然后进行平均或取最大操作。
    if mean:
        result_list = np.mean(result_list, axis=0)
    else:
        result_list = np.max(result_list, axis=0)

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
    result_embedding = []
    # To_Do
    # 第一步：基于consin相似度计算word embedding向量与label embedding之间的相似度
    # 第二步：通过softmax获取注意力分布
    # 第三步：将求得到的注意力分布与输入的word embedding相乘，并对结果进行最大化或求平均
    return result_embedding


def sentence2vec(query, usew2v=True):
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
    if isinstance(query, str):
        query = query.split(" ")
    else:
        raise ValueError("输入格式错误")
    # 加载fast_embedding,w2v_embedding
    global fast_embedding, w2v_embedding
    # fast_arr = np.array([fast_embedding.wv.get_vector(s)
    #                      for s in query if s in fast_embedding.wv.vocab.keys()])
    fast_arr = np.array([w2v_embedding.wv.get_vector(s)
                         for s in query if s in w2v_embedding.wv.vocab.keys()])
    # 在fast_arr下滑动获取到的词向量
    if len(fast_arr) > 0:
        windows_fastarr = np.array(Find_embedding_with_windows(fast_arr))

        result_attention_embedding = Find_Label_embedding(
            fast_arr, fast_embedding)
    else:  # 如果样本中的词都不在字典，则该词向量初始化为0
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
        arr, np.zeros(int(sentence_max_length - len(arr)))))
    return result_arr


def Dimension_Reduction(Train, Test):
    '''
    函数说明：该函数通过PCA算法对样本进行降维，由于目前维度不是特别搞高 ，可以选择不降维。
    参数说明：
    - Train: 表示训练数据集
    - Test: 表示测试数据集
    Return: 返回降维之后的数据样本
    '''
    global max_length
    #  To_Do
    # 特征选择，由于经过特征工程得到的样本表示维度很高，因此需要进行降维 max_length表示降维之后的样本最大的维度。
    # 这里通过PCA方法降维
    pca = PCA(n_components=10, svd_solver='full')
    pca_train = pca.fit_transform(Train)
    pca_test = pca.fit_transform(Test)
    return pca_train, pca_test


def Find_Embedding():
    '''
    函数说明：该函数用于获取经过特征工程之后的样本表示
    Return:训练集特征数组(2D)，测试集特征数组(2D)，训练集标签数组（1D）,测试集标签数组（1D）
    '''
    print("获取样本表示中...")
    min_max_scaler = preprocessing.MinMaxScaler()
    Train_features2 = min_max_scaler.fit_transform(
        np.vstack(train["Text"].apply(sentence2vec)))
    Test_features2 = min_max_scaler.fit_transform(
        np.vstack(test["Text"].apply(sentence2vec)))
    print("获取样本词表示完成")
    # # 通过PCA对样本表示进行降维
    Train_features2, Test_features2 = Dimension_Reduction(
        Train=Train_features2, Test=Test_features2)
    Train_label2 = train["Label"]
    Test_label2 = test["Label"]
    #
    print("加载训练好的词向量")
    print("Train_features.shape =", Train_features2.shape)
    print("Test_features.shape =", Test_features2.shape)
    print("Train_label.shape =", Train_label2.shape)
    print("Test_label.shape =", Test_label2.shape)
    #
    return Train_features2, Test_features2, Train_label2, Test_label2


# Find_Embedding()
def Predict(Train_label, Test_label, Train_predict_label, Test_predict_label, model_name):
    '''
    函数说明：直接输出训练集和测试在模型上的准确率
    参数说明：
        - Train_label: 真实的训练集标签（1D）
        - Test_labelb: 真实的测试集标签（1D）
        - Train_predict_label: 模型在训练集上的预测的标签(1D)
        - Test_predict_label: 模型在测试集上的预测标签（1D）
        - model_name: 表示训练好的模型
    Return: None
    '''
    # ToDo
    # 通过调用metrics.accuracy_score计算训练集和测试集上的准确率
    train_acc = metrics.accuracy_score(Train_label, Train_predict_label)
    test_acc = metrics.accuracy_score(Test_label, Test_predict_label)
    print(model_name + '_' + 'train accuracy %.3f' % train_acc)
    # 输出测试集的准确率
    print(model_name + '_' + 'test accuracy %.3f' % test_acc)


def Grid_Train_model(Train_features, Test_features, Train_label, Test_label):
    '''
    函数说明：基于网格搜索优化的方法搜索模型最优参数，最后保存训练好的模型
    参数说明：
        - Train_features: 训练集特征数组（2D）
        - Test_features: 测试集特征数组（2D）
        - Train_label: 真实的训练集标签 (1D)
        - Test_label: 真实的测试集标签（1D）
    Return: None
    '''
    # parameters = {
    #     'max_depth': [5, 10, 15, 20, 25],
    #     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    #     'n_estimators': [100, 500, 1000, 1500, 2000],
    #     'min_child_weight': [0, 2, 5, 10, 20],
    #     'max_delta_step': [0, 0.2, 0.6, 1, 2],
    #     'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
    #     'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    #     'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    #     'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    #     'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
    # }
    lgb_parameters = {
        'max_depth': [25],
        'learning_rate': [0.1, 0.15],
        'n_estimators': [100, 500],
        'min_child_weight': [5, 10],
        'max_delta_step': [1, 2],
        'subsample': [0.6, 0.7, ],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0.75, 1],
        'reg_lambda': [0.4, 0.6],
        'scale_pos_weight': [0.2, 0.4]
    }
    # 定义分类模型列表，这里仅使用LightGBM模型

    svc_parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100]}
    nb_parameters = {'class_prior': [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]}
    lr_parameters = {'C': [10 ** i for i in range(-4, 4, 2)],  # 指数分布
                     'multi_class': ['ovr', 'multinomial']}
    models = {
        lgb.LGBMClassifier(): lgb_parameters,
        SVC(): svc_parameters,
        MultinomialNB(): nb_parameters,
        LogisticRegression(penalty='l2', solver='lbfgs', tol=1e-6): lr_parameters
    }
    # 遍历模型
    for model, parameters in models.items():
        model_name = model.__class__.__name__
        # gsearch = GridSearchCV(
        #     model, param_grid=parameters, scoring='accuracy', cv=3,n_jobs=-1)
        gsearch = RandomizedSearchCV(
            model, parameters, scoring='accuracy', cv=3, n_jobs=-1)
        gsearch.fit(Train_features, Train_label)
        # 输出最好的参数
        print("Best parameters set found on development set:{}".format(
            gsearch.best_params_))
        Test_predict_label = gsearch.predict(Test_features)
        Train_predict_label = gsearch.predict(Train_features)

        Predict(Train_label, Test_label,
                Train_predict_label, Test_predict_label, model_name)
    # 保存训练好的模型
    # joblib.dump('#ToDo' + '.pkl')


Train_features, Test_features, Train_label, Test_label = Find_Embedding()
Grid_Train_model(Train_features=Train_features, Test_features=Test_features, Train_label=Train_label,
                 Test_label=Test_label)
