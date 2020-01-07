import random
import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
flatten = lambda l: [item for sublist in l for item in sublist]


def getBatch(batch_size, data):
    random.shuffle(data)
    sindex, eindex = 0, 0
    while sindex < len(data):
        batch = data[sindex: eindex]
        sindex, eindex = eindex, eindex + batch_size
        yield batch

    if eindex > len(data):
        batch = data[sindex:]
        yield batch


def prepare_sequence(seq, word2index, unk="<UNK>", return_tensor=False):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index[unk], seq))
    if not return_tensor:
        return idxs
    else:
        return Variable(LongTensor(idxs))


def prepare_word(word, word2index, unk="<UNK>", return_tensor=False):
    idx = word2index.get(word, word2index[unk])
    if not return_tensor:
        return idx
    else:
        return Variable(LongTensor(idx))
