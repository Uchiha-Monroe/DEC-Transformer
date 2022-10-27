
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

import pickle
import random
from tqdm import tqdm


NEG_PERCENT = 0.8

class NormalDataset(Dataset):

    def __init__(self, all_file_list, cluster_label) -> None:
        super(NormalDataset, self).__init__()
        with open(all_file_list, 'rb') as f:
            self.all_file_list = pickle.load(f)
        self.label_np = np.load(cluster_label)
        self.label = list(self.label_np)
    
    def __getitem__(self, index: int):
        '''
        返回格式为 (document， label)
        '''
        return (self.all_file_list[index], self.label[index])

    def __len__(self) -> int:
        assert len(self.all_file_list) == len(self.label)
        return len(self.all_file_list)


class NegDataset(Dataset):

    def __init__(self, all_file_dict, all_file_list) -> None:
        super(NegDataset, self).__init__()
        with open(all_file_dict, 'rb') as f:
            self.all_file_dict = pickle.load(f)
        with open(all_file_list, 'rb') as f:
            self.all_file_list = pickle.load(f)
        # all_file_list长度为9803，共4901组余1，在新数据生成的结尾步骤可能出现问题，故将采样的数据数适当减少以避免
        self.data_len = len(self.all_file_list) // 2 - 51
        
        self.idx_2_clsname = []
        for cls_name in self.all_file_dict.keys():
            self.idx_2_clsname.append(cls_name)
        # 生成新的训练数据
        self.neg_data = []
        for i in tqdm(range(self.data_len)):
            tmp_randint_1 = random.randint(0, 19)
            tmp_random = random.random()
            # 正例样本：两个文档同属同一类别
            if tmp_random > NEG_PERCENT:
                tmp_randint_2 = tmp_randint_1
                doc_1 = self.all_file_dict[self.idx_2_clsname[tmp_randint_1]].pop()
                doc_2 = self.all_file_dict[self.idx_2_clsname[tmp_randint_2]].pop()
                self.neg_data.append = {'doc_1': doc_1, 'doc_2': doc_2, 'label': 1}
            # 负例样本：两个文档属不同类别
            else:
                while 1:
                    tmp_randint_2 = random.randint(0, 19)
                    if tmp_randint_2 != tmp_randint_1 and len(self.all_file_dict[self.idx_2_clsname[tmp_randint_2]]) > 0:
                        break
                doc_1 = self.all_file_dict[self.idx_2_clsname[tmp_randint_1]].pop()
                doc_2 = self.all_file_dict[self.idx_2_clsname[tmp_randint_2]].pop()
                self.neg_data.append = {'doc_1': doc_1, 'doc_2': doc_2, 'label': 0}
    
    def __getitem__(self, index: int):
        '''
        返回数据格式为字典：{'doc_1': doc_1, 'doc_2': doc_2, 'label': 标签值}
        '''
        return self.neg_data[index]

    def __len__(self) -> int:
        return self.data_len



        