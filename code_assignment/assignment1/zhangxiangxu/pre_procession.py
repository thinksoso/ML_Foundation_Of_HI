import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

class Pre_Procession:
    def __init__(self, fname):
        """
        初始化实例，加载数据
        :param fname: 文件路径、名称
        """
        self.train = pd.read_csv(fname, sep='\t')

    def show_info(self):
        """
        展示数据基本信息
        :return: None
        """
        print('train.head:')
        print(self.train.head(), end='\n\n')

        print('train.value_counts:')
        print(self.train.Sentiment.value_counts(), end='\n\n')

        print('train.info:')
        print(self.train.info(), end='\n\n')

        print('train.describe:')
        print(self.train.describe(), end='\n\n')

    def show_count_plot(self):
        """
        展示数据情感信息图表
        :return: None
        """
        plt.figure(figsize=(10, 5))
        ax = plt.axes()
        ax.set_title('分布情况')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        sns.countplot(x=self.train.Sentiment, data=self.train)

    def data_clean(self):
        """
        数据清洗，去除标点，给train增加一列'Phrase1'
        :return: None
        """
        # str.maketrans('','', string.punctuation))使用了三个参数，前两个参数是转换，这里没有使用
        # 第三个参数中出现的字符将被转为None
        self.train['Phrase1'] = self.train.Phrase.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())

