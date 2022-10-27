import pickle
from types import coroutine
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

# 打开经过分词、去停用词等预处理操作后存储在磁盘上的数据集
with open('./all_file_list', 'rb') as f:
    all_file_list = pickle.load(f)
# len_stats_dict用以统计数据集中文本长度
len_stats_dict = OrderedDict([('<=1000', 0), ('1001~2000', 0), ('2001~3000', 0), ('3001~4000', 0), ('4001~5000', 0),
                                ('5001~6000', 0), ('6001~7000', 0), ('7001~8000', 0), ('8001~9000', 0), ('9001~10000', 0), ('>=10000', 0)])

valid_doc_num = 0
length_sum = 0
length_max = 0
length_min = 500
for single_doc in tqdm(all_file_list):
    single_doc = single_doc.replace(' ', '')
    doc_len = len(single_doc)
    if doc_len == 0:
        continue
    else:
        valid_doc_num += 1
    if doc_len > length_max:
        length_max = doc_len
    if doc_len < length_min:
        length_min = doc_len
    length_sum += doc_len
    if doc_len  <= 1000:
        len_stats_dict['<=1000'] += 1
    elif 1000 < doc_len <= 2000:
        len_stats_dict['1001~2000'] += 1
    elif 2000 < doc_len <= 3000:
        len_stats_dict['2001~3000'] += 1
    elif 3000 < doc_len <= 4000:
        len_stats_dict['3001~4000'] += 1
    elif 4000 < doc_len <= 5000:
        len_stats_dict['4001~5000'] += 1
    elif 5000 < doc_len <= 6000:
        len_stats_dict['5001~6000'] += 1
    elif 6000 < doc_len <= 7000:
        len_stats_dict['6001~7000'] += 1
    elif 7000 < doc_len <= 8000:
        len_stats_dict['7001~8000'] += 1
    elif 8000 < doc_len <= 9000:
        len_stats_dict['8001~9000'] += 1
    elif 9000 < doc_len <= 10000:
        len_stats_dict['9001~10000'] += 1
    else:
        len_stats_dict['>=10000'] += 1
length_avg = length_sum / valid_doc_num

print(len_stats_dict)
print(f'the number of raw processed docs is {len(all_file_list)}, the number of valid docs is {valid_doc_num}.')
print(f'the average length of all documents is {length_avg}. \nthe max length is {length_max}. \nthe min length is {length_min}.')

lens_x = list(len_stats_dict.keys())
height_y = list(len_stats_dict.values())
fig, axes = plt.subplots(1, 1, figsize=(15, 6))
axes.bar(lens_x, height_y, width=0.4, label='document length stats', color='#D2ACA3')

axes.set_xticks(list(range(len(len_stats_dict.keys()))))
axes.set_yticks(list(range(0, 2000, 100)))
axes.set_ylim((0, 2000))
axes.set_xticklabels(list(len_stats_dict.keys()))
axes.set_title('document length stats')
axes.grid(linewidth=0.5, which='major', axis='y')

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)

plt.show()