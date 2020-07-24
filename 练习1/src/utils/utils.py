# -*- coding: utf-8 -*-
import jieba
from src.mylib.const import *
import csv
from src.data.data_proc import stopwords
import random
import logging
from logging import handlers


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
        if not text:
            continue
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


def get_corpus(path, tf_idf=False, w2v=False):
    data = csv.reader(open(path, encoding="utf-8"))
    data_list = []
    for n, d in enumerate(data):
        if n == 0:
            continue
        _, text = d
        if tf_idf:
            data_list.append(text)
        elif w2v:
            data_list.append(text.split(" "))
        else:  # dl
            data_list.extend(text.split(" "))
    return data_list


def create_logger(log_path):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger
