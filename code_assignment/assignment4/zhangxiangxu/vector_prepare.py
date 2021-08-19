#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vector_prepare.py    
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/9 21:02   zxx      1.0         None
'''

# import lib
import os

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import keyedvectors
import torch

from dataset import TagDataset

def glo2w2v(source_pth, target_pth):
    glove_file = datapath(source_pth)
    tmp_file = get_tmpfile(target_pth)
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_file, tmp_file)

def getVecs(vector_pth, word2idx, idx2word, embedding_dim=50, cache_dir=None):
    use_cache = os.path.exists(os.path.join(cache_dir, f'{embedding_dim}_embed_cache.pth'))
    if not use_cache:
        wvmodel = keyedvectors.load_word2vec_format(vector_pth, binary=False, encoding='utf-8')
        vocab_size = len(word2idx) + 1
        weight = torch.randn(vocab_size, embedding_dim)

        for i in range(len(wvmodel.index_to_key)):
            try:
                index = word2idx[wvmodel.index_to_key[i]]
            except:
                continue
            weight[index, :] = torch.from_numpy((wvmodel.get_vector(idx2word[str(index)])))
        weight[word2idx['<PAD>'], :] = torch.zeros_like(weight[word2idx['<PAD>'], :])
        vec_cache = os.path.join(cache_dir, f'{embedding_dim}_embed_cache.pth')
        torch.save(weight, vec_cache)
        print("词向量cache存储成功")
    else:
        vec_cache = os.path.join(cache_dir, f'{embedding_dim}_embed_cache.pth')
        weight = torch.load(vec_cache)
        print("词向量读取成功")
    return weight

def main():
    glo2w2v(
        source_pth='D:\documents\word_vector\glove.6B\glove.6B.100d.txt',
        target_pth='D:\documents\word_vector\glove.6B\w2v.100d.txt'
    )

    dataset = TagDataset('conll2003', 'train.txt')
    embeds = getVecs('D:\documents\word_vector\glove.6B\w2v.50d.txt', dataset.word2idx, dataset.idx2word)
    print(embeds[:2])

if __name__ == '__main__':
    main()