#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py    
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/9 19:52   zxx      1.0         None
'''

# import lib
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

import os
import json

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class TagDataset(Dataset):
    def __init__(self, path=None, f_name=None, cache_dir='./cache', from_cache=False, just4Vocab=False):
        super(TagDataset, self).__init__()
        self.path = path
        self.f_name = f_name
        self.cache_dir = cache_dir
        self.START_TAG = "<START>"  # 这两个属于tag
        self.STOP_TAG = "<STOP>"
        self.UNK_TAG = "<UNK>"
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        if just4Vocab:
            self.createVocab(path, f_name)
        else:
            with open(os.path.join(cache_dir, 'unique_vocab_cache.json'), 'r') as jr:
                dic = json.load(jr)
                self.word2idx = dic['word2idx']
                self.idx2word = dic['idx2word']
                self.label2idx = dic['label2idx']
                self.idx2label = dic['idx2label']
            if not from_cache:
                self.data = self._process(path, f_name)
                data_cache = json.dumps(self.data, indent=4)
                with open(os.path.join(cache_dir, f_name.split('.')[0] + '_cache.json'), 'w') as jw:
                    jw.write(data_cache)
            else:
                self._from_cache(cache_dir, f_name)

    def createVocab(self, path, f_name):
        self.word2idx = {'<PAD>': 0}
        self.idx2word = {'0': '<PAD>'}
        word_cnt = 1

        self.label2idx = {'<PAD>': 0}
        self.idx2label = {'0': '<PAD>'}
        label_cnt = 1

        with open(os.path.join(path, f_name), 'r') as fr:
            for line in fr:
                temp = line.strip('\n').lower().split('\t')
                raw_sentence_lst = eval(temp[0])
                word_set = set(raw_sentence_lst)
                for word in word_set:
                    if self.word2idx.get(word, -1) == -1:
                        self.word2idx[word] = word_cnt
                        self.idx2word[str(word_cnt)] = word
                        word_cnt += 1

                raw_label_lst = eval(temp[1])
                label_set = set(raw_label_lst)
                for label in label_set:
                    if self.label2idx.get(label, -1) == -1:
                        self.label2idx[label] = label_cnt
                        self.idx2label[str(label_cnt)] = label
                        label_cnt += 1

        self.vocab_insert(self.START_TAG, self.label2idx, self.idx2label)
        self.vocab_insert(self.STOP_TAG, self.label2idx, self.idx2label)
        self.vocab_insert(self.UNK_TAG, self.word2idx, self.idx2word)

        vocab_cache = json.dumps({
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'label2idx': self.label2idx,
            'idx2label': self.idx2label
        }, indent=4)
        with open(os.path.join(self.cache_dir, 'unique_vocab_cache.json'), 'w') as jw:
            jw.write(vocab_cache)

    def _from_cache(self, cache_dir, f_name):
        with open(os.path.join(cache_dir, f_name.split('.')[0] + '_cache.json'), 'r') as jr:
            self.data = json.load(jr)

    def encode(self, sentence, word2idx):
        return [word2idx[w] for w in sentence]

    def decode(self, idxs, idx2word):
        return [idx2word[str(i)] for i in idxs]


    def vocab_insert(self, new_word, word2idx, idx2word):
        if word2idx.get(new_word, -1) == -1:
            pos = len(word2idx)
            word2idx[new_word] = pos
            idx2word[str(pos)] = new_word
        else:
            raise ValueError("该词已存在")

    def _process(self, path, f_name):
        sentences = []
        labels = []
        lengths = []

        with open(os.path.join(path, f_name), 'r') as fr:
            for line in fr:
                temp = line.strip('\n').lower().split('\t')
                raw_sentence_lst = eval(temp[0])
                sentence_lst = []
                for word in raw_sentence_lst:
                    sentence_lst.append(self.word2idx.get(word, self.word2idx['<UNK>']))

                raw_label_lst = eval(temp[1])
                label_lst = []
                for label in raw_label_lst:
                    label_lst.append(self.label2idx[label])

                sentences.append(sentence_lst)
                labels.append(label_lst)
                lengths.append(len(sentence_lst))
        return {'sentences': sentences, 'labels': labels, 'lengths': lengths}


    def __len__(self):
        assert len(self.data['sentences']) == len(self.data['labels']) == len(self.data['lengths'])
        return len(self.data['labels'])

    def __getitem__(self, index):
        sample = {
            'sentence': self.data['sentences'][index],
            'label': self.data['labels'][index],
            'length': self.data['lengths'][index],
        }
        return sample

def collate_func(batch_dic):
    from torch.nn.utils.rnn import pad_sequence
    batch_len = len(batch_dic)
    max_seq_length = max([dic['length'] for dic in batch_dic])
    mask_batch = torch.zeros((batch_len, max_seq_length)).byte()
    sentence_batch = []
    label_batch = []
    length_batch = []
    for i in range(len(batch_dic)):
        dic = batch_dic[i]
        sentence_batch.append(torch.tensor(dic['sentence'], dtype=torch.long))
        label_batch.append(torch.tensor(dic['label'], dtype=torch.long))
        mask_batch[i, :dic['length']] = 1
        length_batch.append(dic['length'])
    res = {
        'sentence': pad_sequence(sentence_batch, batch_first=True),
        'label': pad_sequence(label_batch, batch_first=True),
        'mask': mask_batch,
        'length': torch.tensor(length_batch, dtype=torch.long)
    }
    return res

if __name__ == '__main__':
    data = TagDataset('conll2003', 'train.txt')
    dataloader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=collate_func)
    for i_batch, batch_data in enumerate(dataloader):
        print(i_batch)
        print(batch_data['sentence'])
        for word in batch_data['sentence'][0]:
            print(data.idx2word[word.item()])
        print(batch_data['label'])
        # print(batch_data['mask'])
        if i_batch > 2:
            break