# -*- coding: utf-8 -*-
import os
import re

LevelLabels = ['强空', '看空', '弱空', '中性', '弱多', '看多', '强多']
Labels2Level = {label: level for level, label in enumerate(LevelLabels)}
save_list = ['train', 'dev', 'test']
curPath = os.path.abspath(os.path.dirname(__file__))

root_path = os.path.split(os.path.split(curPath)[0])[0]

log_dir = root_path + '/log/'

data_path = "data/_data"
data_path = os.path.join(root_path, data_path)
original_data_path = os.path.join(data_path, "manual.csv")
train_data_path = os.path.join(data_path, f"{save_list[0]}.csv")
dev_data_path = os.path.join(data_path, f"{save_list[1]}.csv")
test_data_path = os.path.join(data_path, f"{save_list[2]}.csv")

load_path = "data/dict"
load_path = os.path.join(root_path, load_path)
stocks_path = os.path.join(load_path, "stocks.txt")
stopwords_path = os.path.join(load_path, "stopwords.txt")
han_compile = re.compile(r'[\u4E00-\u9FA5]+')

model_path = "model/save"
model_path = os.path.join(root_path, model_path)
tfidf_path = os.path.join(model_path, "tf_idf.bin")
w2v_path = os.path.join(model_path, "w2v.bin")
fasttext_path = os.path.join(model_path, "fast.bin")
