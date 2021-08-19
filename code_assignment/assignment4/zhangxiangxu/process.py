from tqdm import tqdm
import os
class Conll03Reader:
    def read(self, data_path):
        data_parts = ['_train', '_valid', '_test']
        extension = '.txt'
        dataset = {}
        for data_part in tqdm(data_parts):
            file_path = os.path.join(data_path, data_part+extension)
            dataset[data_part] = self.read_file(str(file_path))
        return dataset

    def read_file(self, file_path):
        samples = []
        tokens = []
        tags = []
        with open(file_path,'r', encoding='utf-8') as fb:
            for line in fb:
                line = line.strip('\n')

                if line == '-DOCSTART- -X- -X- O':
                    # 去除数据头
                    pass
                elif line =='':
                    # 一句话结束
                    if len(tokens) != 0:
                        samples.append((tokens, tags))
                        tokens = []
                        tags = []
                else:
                    # 数据分割，只要开头的词和最后一个实体标注。
                    contents = line.split(' ')
                    tokens.append(contents[0])
                    tags.append(contents[-1])
        return samples
if __name__ == "__main__":
    ds_rd = Conll03Reader()
    data = ds_rd.read("./")
    for key in data.keys():
        with open(str(key)[1:] + '.txt', 'w') as fw:
            for sample in data[key]:
                temp = str(sample[0]) + '\t' + str(sample[1]) + '\n'
                fw.write(temp)
    for sample in data['_train'][:10]:
        print(sample)
