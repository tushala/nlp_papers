# -*- coding: utf-8 -*-

from dataproc import make_dataset
from model import Seq2Seq
from const import *

dataset = make_dataset()
model = Seq2Seq(1111, EMB_DIM, HIDDEN_SIZE, MAX_LENGTH)
for n, (que, ans) in enumerate(dataset):
    