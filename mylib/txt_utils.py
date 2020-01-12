import re
from collections import Counter
from mylib.const import *

re_spaces = re.compile(r"\s+")

intab = "，。“！”"
outtab = r'",."!'
eng_ch_trantab = str.maketrans(intab, outtab)
ch_eng_trantab = str.maketrans(intab, outtab)


def is_chinese(char):
    return '\u4e00' <= char <= '\u9fa5'


def split_sentences(txt):
    sentences = re.split(r"[.。！!?？]", txt)
    return sentences


def chinese_sentences_cut(sentences):
    """中文切词"""
    # todo
    return sentences.split("")


def clean_sentences_to_words(sentences: str, stopwords=False, chinese=False):
    sentences = sentences.strip()
    # todo eng_ch_trantab
    if not chinese:
        words = sentences.split()
        words = [w.lower() for w in words]
    else:
        words = chinese_sentences_cut(sentences)
    if stopwords:
        words = [w for w in words if w not in StopWords]
    return words


def get_corpus(path, chinese_corpus=False):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            words = clean_sentences_to_words(line, True, chinese_corpus)
            corpus.append(words)
    return corpus


def get_vocab(corpus, min_count=10):
    exclude = []
    vocab = Counter(flatten(corpus))
    for w, c in vocab.items():
        if c > min_count:
            exclude.append(w)
    return exclude
