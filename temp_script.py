
import os
import re
import numpy
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np
import torch

from typing import List



# with open('./all_file_dict_test', 'rb') as f:
#     all_file_dict = pickle.load(f)

# all_file_label = []
# cls_index = 0
# for cls_name in all_file_dict:
#     all_file_label.extend([cls_index] * len(all_file_dict[cls_name]))
#     cls_index += 1
# print(len(all_file_label))

# all_label_ndarray = numpy.array(all_file_label)
# numpy.save('./cluster_label_test.npy', all_label_ndarray)

tuned_result = torch.load('./fine_tune/fine_tuned_result.pt')

labels_np = np.load('./cluster_label_test.npy')
labels = torch.tensor(labels_np)

ratio = (tuned_result == labels).to(dtype=int).sum().item() / len(labels)
print(f'准确率是：{round(ratio * 100, 2)}%')