"""负采样 原文中没有体现 负采样和Hs同为优化方法负采样方法更为喜闻乐见"""
import mylib.word_tools as wt
import mylib.txt_utils as tu
from mylib import const
from fasttext.utils import get_trian_data
import nltk
WINDOW_SIZE = 3
PAD = '<DUMMY>'
data_path = r"C:\Users\19416\AppData\Roaming\nltk_data\corpora\gutenberg\melville-moby_dick.txt"

corpus = tu.get_corpus(data_path, False)

vocab_list = tu.get_vocab(corpus)
word2index, index2word = wt.get_wordindex(vocab_list)
windows = const.flatten([list(nltk.ngrams([PAD] * WINDOW_SIZE + c + [PAD] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])
train_data = get_trian_data(windows, WINDOW_SIZE)
print(windows)