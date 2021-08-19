#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/18 19:33   zxx      1.0         None
@reference: https://zhuanlan.zhihu.com/p/5658208
'''

# import lib
class Evaluator:
    def __init__(self, idx2label):
        self.idx2label = idx2label

    def _decode(self, idxs):
        if not isinstance(idxs, list):
            idxs = idxs.tolist()
        return [self.idx2label[str(i)] for i in idxs[0]]

    def _get_entity(self, idxs):
        entity_set = {}
        entity_pointer = None
        label_seq = self._decode(idxs)
        for i, label in enumerate(label_seq):
            if label.startswith('b'):
                category = label.split('-')[1]
                entity_pointer = (i, category)
                entity_set.setdefault(entity_pointer, [label])
            elif label.startswith('i'):
                if entity_pointer is None: continue
                if entity_pointer[1] != label.split('-')[1]: continue
                entity_set[entity_pointer].append(label)
            else:
                entity_pointer = None
        return entity_set

    def compute_F1(self, pred, target):
        real_entiy_set = self._get_entity(target)
        pred_entiy_set = self._get_entity(pred)

        pred_true_entity_set = {}
        total_keys = real_entiy_set.keys() & pred_entiy_set.keys()

        for key in total_keys:
            real_label = real_entiy_set.get(key)
            pred_label = pred_entiy_set.get(key)

            if tuple(real_label) == tuple(pred_label):
                pred_true_entity_set.setdefault(key, real_label)

        TP_add_FP = len(pred_entiy_set)
        TP = len(pred_true_entity_set)
        TP_add_FN = len(real_entiy_set)

        if TP_add_FP != 0:
            precision = TP / TP_add_FP
        else:
            precision = 0

        if TP_add_FN != 0:
            recall = TP / TP_add_FN
        else:
            recall = 0

        if recall + precision != 0:
            F1 = 2 * recall * precision / (recall + precision)
        else:
            F1 = 0

        return precision, recall, F1