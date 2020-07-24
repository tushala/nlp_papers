import jieba
from src.mylib.const import *

stopwords = None


def load_stopwords(path):
    global stopwords
    if stopwords is not None:
        return stopwords
    stopwords = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords


def init_start():
    jieba.load_userdict(stocks_path)
    load_stopwords(stopwords_path)



