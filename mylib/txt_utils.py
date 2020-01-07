import re
from collections import Counter
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

def clean_sentences_to_words(sentences:str, stopwords=False, chinese=False):
    sentences = sentences.strip()
    if stopwords
def get_corpus(path, chinese_corpus=False):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            words = clean_sentences_to_words(line)
            corpus.extend(words)
    return Counter(corpus)