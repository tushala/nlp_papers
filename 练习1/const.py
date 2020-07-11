# -*- coding: utf-8 -*-
import os
import re
LevelLabels = ['强空', '看空', '弱空', '中性', '弱多', '看多', '强多']
Labels2Level = {label: level for level, label in enumerate(LevelLabels)}
save_list = ['train', 'dev', 'test']

data_path = "data/_data"
original_data_path = os.path.join(data_path, "manual.csv")
train_data_path = os.path.join(data_path, f"{save_list[0]}.csv")
dev_data_path = os.path.join(data_path, f"{save_list[1]}.csv")
test_data_path = os.path.join(data_path, f"{save_list[2]}.csv")

load_path = "data/dict"
stocks_path = os.path.join(load_path, "stocks.txt")
stopwords_path = os.path.join(load_path, "stopwords.txt")
han_compile = re.compile(r'[\u4E00-\u9FA5]+')
