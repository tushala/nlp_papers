# -*- coding: utf-8 -*-
import torch
USE_CUDA = torch.cuda.is_available()
WINDOW_SIZE = 3
EPOCH = 10
corpus_txt = 'melville-moby_dick.txt'
BATCH_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 10