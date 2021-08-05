import pandas as pd
import numpy as np


def function(x, n):
    """
    用于create_couple_dictionary生成词组
    """
    temp = [x.split()[i: i + n] for i in range(len(x.split()) - n + 1)]
    res = tuple(t for t in set(tuple(_) for _ in temp))
    return res





class Representation:
    def __init__(self, df):
        """
        需要读取一个DataFrame
        :param df:
        """
        self.df = df

        self.word_matrix = None
        self.word_to_index = {}
        self.index_to_word = {}

        self.couple_matrix = None
        self.couple_to_index = {}
        self.index_to_couple = {}

    def convert_to_one_hot(self, c=5):
        """
        输入一个series，内容为整数，转为独热向量矩阵，维度为c*样本数
        参数：
        self.df.sentiment：df中包含Sentiment的一列
        c：类别数，int
        返回：
        Y：独热向量矩阵，c*样本数
        """
        num = self.df.sentiment.shape[0]
        Y = np.eye(c)[np.array(self.sentiment).reshape(-1)].T
        assert Y.shape == (c, num)
        return Y

    def create_dictionary(self):
        """
        输入数据self.df，建立词袋，给数据df增添一个新的列Set_of_word，里面包含该样本中所含有的词汇，同时返回一个包含所有词汇的列表
        """
        self.df['Set_of_word'] = self.df.Phrase1.apply(lambda x: set(x.split()))
        bag_of_word = set()
        for _, value in self.df.Set_of_word.items():
            bag_of_word |= value
        return list(bag_of_word)

    def create_couple_dictionary(self, n=2):
        """
        输入数据df，建立词组袋，给数据df增添一个新的列Set_of_word_couple，里面包含该样本中所含有的词组，同时返回一个包含所有词组的列表
        参数：
        self.df：DataFrame, 大小为156060*5，包含所有数据
        n：词组的个数
        """
        self.df['Set_of_word_couple'] = self.df.Phrase1.apply(function, n=n)
        bag_of_word_couple = set()
        for _, value in self.df.Set_of_word_couple.items():
            bag_of_word_couple |= set(value)
        return list(bag_of_word_couple)

    def bag_of_word(self):
        """
        根据数据表格，生成包含所有词的列表,对于每行数据，根据其中包含的词构建词袋向量
        参数：
        self.df：DataFrame, 大小为156060*5，包含所有数据，至少有五列：PhraseId，SentenceId，Phrase，Sentiment，Phrase1，运行结束后增加一列Set_of_word
        返回：
        self.word_matrix：包含词袋向量的矩阵，维度为词汇量*样本量
        self.word_to_index：一个词对→索引的字典
        self.index_to_word：一个索引→词对的字典
        """
        amount, _ = self.df.shape
        bag_of_word = create_dictionary(self.df)  # 词袋列表，同时给self.df增加一个新的列Set_of_word
        dimension = len(bag_of_word)
        # 设置一个矩阵，用来盛放每条数据的词袋向量，顺序与self.df一致，大小为词汇数量*样本数量
        self.word_matrix = np.zeros((dimension, amount), dtype='i1')
        # 遍历数据，填充self.word_matrix，对于每一行的样本，先将其按照单词拆为列表，然后找到每个单词在self.word_matrix中的位置，将其数值加1
        self.word_to_index = {}  # 一个词汇→索引的字典
        self.index_to_word = {}  # 一个索引→词汇的字典
        for row in self.df.itertuples():
            index = getattr(row, 'Index')
            temp_list = getattr(row, 'Phrase1').split()  # 得到的是列表类型
            for word in temp_list:
                # 确定单词在词袋中的位置
                if self.word_to_index.get(word, -1) < 0:
                    position = bag_of_word.index(word)
                    self.word_to_index[word] = position
                    self.index_to_word[position] = word
                else:
                    position = self.word_to_index.get(word, -1)
                # 修改self.word_matrix
                self.word_matrix[position][index] += 1
        assert self.word_matrix.shape == (len(bag_of_word), self.df.shape[0])
        return self.word_matrix, self.word_to_index, self.index_to_word

    def n_gram(self, n):
        """
         根据数据表格，生成包含所有词组的列表,对于每行数据，根据其中包含的词组构建词袋向量
         参数：
         self.df：DataFrame, 大小为156060*5，包含所有数据，至少有五列：PhraseId，SentenceId，Phrase，Sentiment，Phrase1，运行结束后增加一列Set_of_word_couple
         n：每个组合中包含词的数量
         返回：
         self.couple_matrix：包含词袋向量的矩阵，维度为词汇量*样本量
         self.couple_to_index：一个词对→索引的字典
         self.index_to_couple：一个索引→词对的字典
        """
        amount, _ = self.df.shape
        bag_of_word = create_couple_dictionary(self.df, n=2)  # 词袋列表，同时给self.df增加一个新的列Set_of_word
        dimension = len(bag_of_word)
        # 设置一个矩阵，用来盛放每条数据的词袋向量，顺序与self.df一致，大小为词汇数量*样本数量
        self.couple_matrix = np.zeros((dimension, amount), dtype='i1')
        # 遍历数据，填充self.couple_matrix，对于每一行的样本，先将其按照单词拆为列表，然后找到每个单词在self.couple_matrix中的位置，将其数值加1
        self.couple_to_index = {}  # 一个词汇→索引的字典
        self.index_to_couple = {}  # 一个索引→词汇的字典
        for row in self.df.itertuples():
            index = getattr(row, 'Index')
            temp = getattr(row, 'Phrase1').split()
            temp_list = [tuple(temp[i: i + n]) for i in range(len(temp) - n + 1)]  # 得到的是列表类型
            for couple in temp_list:
                # 确定单词在词袋中的位置
                if self.couple_to_index.get(couple, -1) < 0:
                    position = bag_of_word.index(couple)
                    self.couple_to_index[couple] = position
                    self.index_to_couple[position] = couple
                else:
                    position = self.couple_to_index.get(couple, -1)
                # 修改self.couple_matrix
                self.couple_matrix[position][index] += 1
            if index % 100 == 0:
                print(f'{index / amount * 100:.2f}%', end='\r')
        assert self.couple_matrix.shape == (len(bag_of_word), self.df.shape[0])
        return self.couple_matrix, self.couple_to_index, self.index_to_couple

    def feature_select(self, choice = 'word', threshold=.8 * (1 - .8)):
        """
        删除方差较小的特征
        参数：
        choice:特征选择的对象，word针对word_matrix,couple针对couple_matrix
        threshold：标量，阈值
        返回：
        X_selected：array,选择后的特征
        idx: 删除的下标
        """
        X = None
        if choice == 'word':
            X = self.word_matrix
        elif choice == 'couple':
            X = self.couple_matrix
        else:
            print('请输入正确的选择名称')

        d, n = X.shape
        var = np.var(X, axis=1).reshape((d, 1))
        mask = var > threshold
        temp = mask * X
        idx = np.argwhere(np.all(temp[..., :] == 0, axis=1))
        X_selected = np.delete(X, idx, axis=0)

        return X_selected, idx



