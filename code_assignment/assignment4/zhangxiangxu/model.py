#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py    
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/17 21:42   zxx      1.0         None
'''

# import lib
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hiddens_size, num_layers, dropout_prob, tagset_size, device):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hiddens_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.tagset_size = tagset_size
        self.device = device

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.BiLSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hiddens_size // 2, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hiddens_size, self.tagset_size)
        self.BN = nn.BatchNorm1d(self.tagset_size)

        self.hidden = None

    def init_hidden(self, batch_size):
        return torch.randn(2 * self.num_layers, batch_size, self.hidden_size // 2, device=self.device), torch.randn(2 * self.num_layers, batch_size, self.hidden_size // 2, device=self.device)

    def forward(self, sentence):
        self.hidden = self.init_hidden(sentence.shape[0])
        embeds = self.embedding(sentence)
        embeds = self.dropout(embeds)
        lstm_out, self.hidden = self.BiLSTM(embeds, self.hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_features = self.hidden2tag(lstm_out)
        return self.BN(lstm_features.transpose(1, 2)).transpose(1, 2)


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hiddens_size, num_layers, dropout_prob, tagset_size, device):
        super(BiLSTM_CRF, self).__init__()
        self.BiLSTM = BiLSTM(vocab_size, embedding_dim, hiddens_size, num_layers, dropout_prob, tagset_size, device)
        self.CRF = CRF(num_tags=tagset_size, batch_first=True)

    def forward(self, sentence, tags, masks):
        feats = self.BiLSTM(sentence)
        loss = self.CRF(feats, tags, mask=masks)
        return loss

    def decode(self, sentence):
        feats = self.BiLSTM(sentence)
        return self.CRF.decode(feats)