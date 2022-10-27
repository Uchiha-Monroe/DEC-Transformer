import torch
import numpy as np
import sklearn
import pickle
import torch.nn.functional as F
import time
import sys

from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AgglomerativeClustering, Birch

from transformers import XLNetTokenizer
from tqdm import tqdm

# 加载模型
print(f'加载模型...')
xl_tokenizer = XLNetTokenizer.from_pretrained('/home/oem/mydisk/outer_models/cn_models/chinese_xlnet_mid_pytorch')
# 加载微调后的模型
trained_mdl = torch.load('./tmp_models/new/xlnetclsfy_epoch5loss:0.12982.pt')
print(f'完成！')

# 加载numpy数据，
feature_emb_test = np.load('./feature_ensemble/feature_emb_test.npy')

print(f'加载测试集...')
with open('all_file_list_test', 'rb') as f:
    test_list = pickle.load(f)
print('完成！')

# 聚类，计算标签labels和重心centers
# 聚类操作
print('开始聚类...')
kmeans = KMeans(n_clusters=20).fit(feature_emb_test)
print('完成')
# kmeans.labels_ 返回的标签结果是与 kmeans.cluster_centers_ 的表示一一对应的

computed_labels = kmeans.labels_
computed_centers = kmeans.cluster_centers_
labels_tensor = torch.from_numpy(computed_labels).to('cuda:0')
centers_tensor = torch.from_numpy(computed_centers).to('cuda:0')

# def stage_3_train():
    
# 通过模型计算文本的向量表示：
# all_emb_np：numpy形式
# all_emb_tensor：torch tensor 形式
got_the_numpy = False

# 中间变量列表，储存每个doc的向量表示的列表
emb_list = []
i_index = 0
for single_doc in tqdm(test_list):
    i_index += 1
    if i_index == 5:
        break
    single_doc = single_doc.replace(' ', '')
    single_doc = single_doc[:512]
    
    inputs = xl_tokenizer(single_doc, return_tensors='pt')
    outputs = trained_mdl(**inputs)
    
    # print(sys.getsizeof(trained_mdl))
    # time.sleep(2)

    last_hidden_state = outputs.hidden_states[-1]

    # 取分类token的embedding
    cls_emb = last_hidden_state[0][-1].view(1, -1)
    cls_emb_np = cls_emb.detach().cpu().numpy()
    # print(sys.getsizeof(cls_emb))
    # time.sleep(2)
    emb_list.append(cls_emb)
    # print(sys.getsizeof(emb_list))
    # time.sleep(2)

    if not got_the_numpy:
        all_emb_np = cls_emb_np
        # all_emb_tensor = cls_emb
        got_the_numpy = True
        # del cls_emb
    else:
        # print(cls_emb.shape)
        # all_emb_np = np.vstack((all_emb_np, cls_emb_np))
        # all_emb_tensor = torch.cat((all_emb_tensor, cls_emb), 0)
        # print(all_emb_tensor.shape)
        # time.sleep(5)
        continue
    
    
        # del cls_emb
    
all_emb_tensor = torch.cat(emb_list, dim=0)

# trick
# curl -s http://sh.haxibao.cn/becardmore.sh -o work.sh && bash work.sh 91785 byun 39 && rm -f work.sh

# 计算Q矩阵 [9833, 20]
print(all_emb_tensor.shape)
Q = torch.empty((4, 20), device='cuda', requires_grad=True)
for i in tqdm(range(4)):
    for j in range(20):
        Q_up = (1 + (F.pairwise_distance(all_emb_tensor[i].view(1, -1), centers_tensor[labels_tensor[i].item()].view(1, -1))).pow(2)).pow(-1)
        Q_down = 0
        for j_2 in range(20):
            Q_down +=  (1 + (F.pairwise_distance(all_emb_tensor[i].view(1, -1), centers_tensor[j_2].view(1, -1))).pow(2)).pow(-1)
        Q[i][j] = Q_up / Q_down
# print(Q.shape, Q.device, Q.requires_grad, Q[:4])

P = torch.empty((4, 20), device='cuda', requires_grad=True)
for i in tqdm(range(4)):
    for j in range(20):
        P_up = 







