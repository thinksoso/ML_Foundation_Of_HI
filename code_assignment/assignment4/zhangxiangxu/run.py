#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      :   run_.py
@Contact   :   xxzhang16@fudan.edu.cn
@reference :   https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
               https://zhuanlan.zhihu.com/p/42096344
               https://blog.csdn.net/Tangjing_Pacino/article/details/106620479
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/4 19:28   zxx      1.0         None
'''

# import lib

import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
import os
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import BiLSTM_CRF
from dataset import TagDataset, collate_func
from vector_prepare import getVecs
from utils import Evaluator

DEVICE = 'cuda'
def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for i, sample in enumerate(tqdm(dataloader)):
        model.zero_grad()

        sentence_in = sample['sentence'].to(DEVICE)
        targets = sample['label'].to(DEVICE)
        masks = sample['mask'].to(DEVICE)

        loss = -model(sentence_in, targets, masks)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, evaluator):
    model.eval()
    precision = 0
    recall = 0
    F1 = 0
    n = len(dataloader)
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataloader)): # val_loader batch大小为1
            sentence_in = sample['sentence'].to(DEVICE)
            target = sample['label'].to(DEVICE)
            pred = model.decode(sentence_in)
            _precision, _recall, _F1 = evaluator.compute_F1(pred, target)
            precision += _precision
            recall += _recall
            F1 += _F1
    return precision / n, recall / n, F1 / n

def main():
    config = {
        "EMBEDDING_DIM" : 100,
        "HIDDEN_DIM" : 512,
        "DROPOUT_PROB" : 0.5,
        "BATCH_SIZE" : 32,
        "NUM_WORKERS" : 8,
        "NUM_LAYERS": 1,
        "EPOCH" : 50,
        "LEARNING_RATE" : 1e-3,
        "WEIGHT_DECAY" : 1e-3,
        "CACHE_DIR" : "./cache",
        "DATA_DIR" : 'conll2003',
        "SET_FOR_VACAB" : 'bind.txt',
        "TRAIN_SET_FN" : 'train.txt',
        "EVAL_SET_FN" : 'valid.txt',
        "TEST_SET_FN" : 'test.txt',
        "FROM_CACHE" : False,
        "EVAL_STEP" : 5,
    }

    EMBEDDING_DIM = config["EMBEDDING_DIM"]
    HIDDEN_DIM = config["HIDDEN_DIM"]
    DROPOUT_PROB = config["DROPOUT_PROB"]
    NUM_LAYERS = config["NUM_LAYERS"]
    NUM_WORKERS = config["NUM_WORKERS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    EPOCH = config["EPOCH"]
    LEARNING_RATE = config["LEARNING_RATE"]
    WEIGHT_DECAY = config["WEIGHT_DECAY"]
    CACHE_DIR = config["CACHE_DIR"]
    DATA_DIR = config["DATA_DIR"]
    SET_FOR_VACAB = config["SET_FOR_VACAB"]
    TRAIN_SET_FN = config["TRAIN_SET_FN"]
    EVAL_SET_FN = config["EVAL_SET_FN"]
    TEST_SET_FN = config["TEST_SET_FN"]
    FROM_CACHE = config["FROM_CACHE"]
    EVAL_STEP = config["EVAL_STEP"]
    TIME = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())

    with open(f'./log/log_text/{TIME}.txt', 'w') as fw:
        for key in config.keys():
            fw.write(key + " " + str(config[key]) + '\n')
        fw.write('\n\n\n')
    cache_exist = os.path.exists(CACHE_DIR + '/unique_vocab_cache.json') and \
                  os.path.exists(CACHE_DIR + '/' + TRAIN_SET_FN.split('.')[0] + '_cache.json') and \
                  os.path.exists(CACHE_DIR + '/' + EVAL_SET_FN.split('.')[0] + '_cache.json') and \
                  os.path.exists(CACHE_DIR + '/' + TEST_SET_FN.split('.')[0] + '_cache.json')
    if cache_exist:
        FROM_CACHE = True
        train_data = TagDataset(DATA_DIR, TRAIN_SET_FN, cache_dir=CACHE_DIR, from_cache=FROM_CACHE, just4Vocab=False)
        eval_data = TagDataset(DATA_DIR, EVAL_SET_FN, cache_dir=CACHE_DIR, from_cache=FROM_CACHE, just4Vocab=False)
        test_data = TagDataset(DATA_DIR, TEST_SET_FN, cache_dir=CACHE_DIR, from_cache=FROM_CACHE, just4Vocab=False)
        print("数据读取自cache")
    else:
        TagDataset(DATA_DIR, SET_FOR_VACAB, from_cache=FROM_CACHE, just4Vocab=True)
        train_data = TagDataset(DATA_DIR, TRAIN_SET_FN, cache_dir=CACHE_DIR, from_cache=FROM_CACHE, just4Vocab=False)
        eval_data = TagDataset(DATA_DIR, EVAL_SET_FN, cache_dir=CACHE_DIR, from_cache=FROM_CACHE, just4Vocab=False)
        test_data = TagDataset(DATA_DIR, TEST_SET_FN, cache_dir=CACHE_DIR, from_cache=FROM_CACHE, just4Vocab=False)
        print("cache创建完成")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func, num_workers=NUM_WORKERS, pin_memory=True)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, collate_fn=collate_func, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_func, num_workers=NUM_WORKERS, pin_memory=True)
    embeds = getVecs(f'D:\documents\word_vector\glove.6B\w2v.{EMBEDDING_DIM}d.txt', train_data.word2idx, train_data.idx2word, 100, cache_dir=CACHE_DIR)

    word_to_ix = train_data.word2idx
    tag_to_ix = train_data.label2idx
    # 创建模型和优化函数
    model = BiLSTM_CRF(len(word_to_ix), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT_PROB, len(tag_to_ix), DEVICE)
    model.BiLSTM.embedding.from_pretrained(embeds, freeze=False, padding_idx=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    evaluator = Evaluator(train_data.idx2label)

    # 训练
    model.to(DEVICE)
    for epoch in range(EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data
        train_loss = train(model, train_loader, optimizer)
        train_log = f"EPOCH: {epoch}\ttrain loss: {train_loss:.4f}\n"
        eval_log = ""
        if epoch % EVAL_STEP == 0:
            precision, recall, F1 = evaluate(model, eval_loader, evaluator)
            eval_log = f"EPOCH: {epoch}\tevaluate precision: {precision:.2%}\t" \
                   f"recall: {recall:.2%}\tF1: {F1:.2%}\n"
            with SummaryWriter(log_dir=f'./log/log_graph/{TIME}', comment='train') as writer:
                writer.add_scalars('data/loss', {"train_loss": train_loss}, epoch)
            with SummaryWriter(log_dir=f'./log/log_graph/{TIME}', comment='evaluate') as writer:
                writer.add_scalars('data/metric', {'precision': precision, 'recall': recall, 'F1': F1}, epoch)

        total_log = train_log + eval_log
        print(total_log, end='')
        with open(f'./log/log_text/{TIME}.txt', 'a') as fw:
            fw.write(total_log)

    precision, recall, F1 = evaluate(model, test_loader, evaluator)
    test_log = f"\nFinal Score\ntest precision: {precision:.2%}\t" \
               f"recall: {recall:.2%}\tF1: {F1:.2%}\n"
    print(test_log)
    with open(f'./log/log_text/{TIME}.txt', 'a') as fw:
        fw.write(test_log)

if __name__ == '__main__':
    main()