import torch
from nltk.corpus import stopwords
USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
flatten = lambda l: [item for sublist in l for item in sublist]


Eng_StopWords = set(stopwords.words("english"))
Chinese_StopWords = set()  # todo 需要中英文停用词表

StopWords = Eng_StopWords | Chinese_StopWords
