from get_data import Dataset, WINDOW_SIZE
import os
import torch
from torch import nn
from new_model import model
from torch import optim
from sklearn.metrics import accuracy_score

SENTENCE_LENGTH = WINDOW_SIZE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EMBED_SIZE = 400
data = Dataset()
vocab_size = len(data.word2id)
# SAVE_PATH = r'data/model/train/best.pth'
# Test_path = r'data/model/test/best.pth'
# mlp_path = r'data/model/mlp/best.pth'
SAVE_PATH = r'data/model/best.pth'
label_size = 30000

EPOCH = 5
BATCH_SIZE = 512
USE_CUDA = torch.cuda.is_available()


def l2_regular(model:nn.Module, alpha):
    loss = torch.Tensor([0]).cuda()
    for param in model.parameters():
        loss += torch.norm(param)
    return (loss / 2) * alpha


def train(mode):
    # 先训练训练集的词向量和句向量
    # 句子 + 文章id --> 下一个词
    lr = 2e-3
    loss_function = nn.CrossEntropyLoss()
    
    m = model(vocab_size, EMBED_SIZE, SENTENCE_LENGTH)
    if mode == 'test':
        m.load_state_dict(torch.load(SAVE_PATH))
    if USE_CUDA:
        m = m.cuda()
    m.train()
    optimizer = optim.Adam(m.parameters(), lr=lr)
    train_length = len(data.train_word_datas)
    test_length = len(data.test_word_datas)
    best_loss = float('inf')
    for epoch in range(EPOCH):
        step = 0
        start = 0
        end = start + BATCH_SIZE
        length = train_length if mode == 'train' else test_length
        while end < length:
            step += 1
            optimizer.zero_grad()
            train_word_set = torch.LongTensor(data.train_word_datas[start:end]) if mode == 'train' \
                else torch.LongTensor(data.test_word_datas[start:end])  # 512 * 9
            train_para_set = torch.LongTensor(data.train_para_datas[start:end]) if mode == 'train' \
                else torch.LongTensor(data.test_para_datas[start:end])  # 512 * 1
            train_label_set = torch.LongTensor(data.train_new_labels[start:end]) if mode == 'train' \
                else torch.LongTensor(data.test_new_labels[start:end])  # 512
            if USE_CUDA:
                train_word_set = train_word_set.cuda()
                train_para_set = train_para_set.cuda()
                train_label_set = train_label_set.cuda()
            
            out = m(train_word_set, train_para_set, train_label_set)
            neg = l2_regular(m, 1e-5)
            loss = loss_function(out, train_label_set) + neg
            loss.backward()
            
            if step % 1000 == 0:
                print(start)
            start = end
            end = min(end + BATCH_SIZE, length)
            optimizer.step()
        lr *= 0.99
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss.item()))
        # torch.save(model.state_dict(), 'data/model/train/{}.pth'.format(epoch))
        if loss.item() < best_loss:
            torch.save(m.state_dict(), SAVE_PATH)


def mlp(mode):
    # 固定词向量和句向量，开始训练单隐层神经网络分类器用于情感分类
    # 文章id --> 感情输出
    lr = 2e-3
    if not os.path.exists(SAVE_PATH):
        raise Exception("路径不存在")
    loss_function = nn.CrossEntropyLoss()
    m = model(vocab_size, EMBED_SIZE, SENTENCE_LENGTH)
    m.load_state_dict(torch.load(SAVE_PATH))
    optimizer = optim.Adam(m.parameters(), lr=lr)
    length = len(data.train_labels) if mode == 'train' else len(data.test_labels)  # 25000
    para_datas = torch.arange(0, length + 1).view(-1, 1).long()
    mlp = data.train_labels if mode == 'train' else data.test_labels
    data_labels = data.train_labels if mode == 'train' else data.test_labels
    if USE_CUDA:
        para_datas = para_datas.cuda()
        data_labels = torch.LongTensor(data_labels).cuda()
        m = m.cuda()
    epoch = 10 if mode == 'train' else 1
    for e in range(epoch):
        pred = []
        start = 0
        end = start + BATCH_SIZE
        if mode == 'train':
            while start < length * 0.8:
                optimizer.zero_grad()
                train_para_set = para_datas[start:end]  # 512 * 1
                train_article_label = data_labels[start:end]  # 512
                
                out = m.mlp(train_para_set)
                loss = loss_function(out, train_article_label)
                loss.backward()
                optimizer.step()
                start = end
                end = min(end + BATCH_SIZE, 20000)
                lr *= 0.99
            print('Epoch:', '%04d' % (e + 1), 'cost =', '{:.6f}'.format(loss.item()))
            end = start + BATCH_SIZE
            while start < length:
                train_para_set = para_datas[start:end]  # 512 * 1
                out = m.mlp(train_para_set)
                pre = torch.argmax(out, 1)
                pred.extend(pre.tolist())
                start = end
                end = min(end + BATCH_SIZE, length)
            acc = accuracy_score(pred, mlp[20000:])
            print('Epoch:', e, '  mlp acc:', acc)
            torch.save(m.state_dict(), SAVE_PATH)
        else:
            while start < length:
                test_para_set = para_datas[start:end]
                out = m.mlp(test_para_set)
                pre = torch.argmax(out, 1)
                pred.extend(pre.tolist())
                start = end
                end = min(end + BATCH_SIZE, length)
            acc = accuracy_score(pred, mlp)
            print('test mlp acc:', acc)


if __name__ == '__main__':
    train()
    mlp('train')
    train('test')
    mlp('test')
