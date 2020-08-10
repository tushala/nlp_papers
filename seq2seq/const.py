# -*- coding: utf-8 -*-
import torch
MAX_LENGTH = 20
BATCH_SIZE = 10
EMB_DIM = 300
HIDDEN_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")