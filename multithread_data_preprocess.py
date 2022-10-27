import os
import re
from tqdm import tqdm
from multiprocessing import Pool

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np

from typing import List

import jieba


# def read_files(path:str, all_files=[], all_file_dict={}, all_file_list=[]):
#     file_list = os.listdir(path)
#     all_file_dict = {}
#     for file in file_list:
#         all_file_dict[file] = []
#     # print(all_file_dic)
#     for file in file_list:
#         cur_path = os.path.join(path, file)
#         print(cur_path)
#         for cur_tmp_path in cur_path:
#             new_path = os.path.join(cur_path, cur_tmp_path)
#             print(new_path)
#             for cur_file in new_path:
#                 print(cur_file)
#                 all_file_dict[file].append(read_preprocess_file(cur_file))
#     return all_file_dict


def show_files(path, all_files):
    # 返回所有txt文件路径组成的列表
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            all_files.append(cur_path)
    return all_files


def read_preprocess_file(filename):
    # 读取.txt文件中的内容，只保留中文字符，对提取的文本进行分词，去停用词，最终将处理后的文本输出
    # input:.txt文件的绝对路径
    # output:字符串
    # 匹配中文字符
    cop = re.compile("[^\u4e00-\u9fa5]")
    with open(filename, encoding='utf-8') as f:
        data = ''
        for line in f.readlines():
            line = line.strip()
            data += line
            new_data = cop.sub('', data)
            
            stopwords = ['的','了','但是','况且','不仅','还','为了','主要','正文','作者','关键词','称之为',
                            '认为','因此','人们','一个','需要','我们','你们','他们','这种','标题','进一步','如图所示',
                            '一种','一定','由于','条件','进行','作为','一个','可以','mm','原文','更进一步','文献号',
                            'xi','结果','版号','标题','日期','合作','去年','今年','目前','期号','作者简介','哲社版',
                            '昨天','一直','作者','存在','正文','形成','要求','产生','因为','因此','期号']
            wordlist = jieba.lcut(new_data)
            wordlist = [w for w in wordlist if w not in stopwords ]
            document = ' '.join(wordlist)
    return document


def create_filecls_dict(path='./train'):
    # 创建包含文本主题类的字典
    file_list = os.listdir(path)
    all_file_dict = {}
    cls_lst = []
    for file in file_list:
        all_file_dict[file] = []
    return all_file_dict


def dump_txt_2_dict(filename):
    # 根据输入的.txt文件路径名返回其所属分类的字符
    
    tmp_lst = filename.split('/')
    pure_filename = tmp_lst[-1]
    txt_pat = re.compile(r'C\d+-[\D]*')
    m = txt_pat.search(pure_filename)
    return m.group()



all_files = show_files('./train', [])

# all_file_dict：按类别保存所有预处理和清洗后的文档列表
all_file_dict = create_filecls_dict()
print(all_file_dict)


def dump_and_prepro(filename):
    cls_name = dump_txt_2_dict(filename)
    target_doc = read_preprocess_file(filename)
    all_file_dict[cls_name].append(target_doc)


pool = Pool(20)

for single_file in tqdm(all_files):
    
    pool.apply_async(dump_and_prepro(single_file))
    

pool.close()
pool.join()




all_file_list = []
for cls_name in all_file_dict:
    all_file_list.extend(all_file_dict[cls_name])
print(len(all_file_list))

vectorizer = TfidfVectorizer()
all_file_list = vectorizer.fit_transform(all_file_list)
all_file_array = np.array(all_file_list)
kmeans = KMeans(n_clusters=20, random_state=0)
print(kmeans.labels_)





# print(all_files)
# tmp_doc = read_preprocess_file(all_files[1999])
# print(tmp_doc)
    
# last_dict = read_files(path='./train')
# print(last_dict)            

