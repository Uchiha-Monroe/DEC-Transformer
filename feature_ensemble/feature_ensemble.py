import torch
import pickle
import numpy as np

from transformers import XLNetTokenizer
from tqdm import tqdm

print(f'加载模型...')
xl_tokenizer = XLNetTokenizer.from_pretrained('/home/oem/mydisk/outer_models/cn_models/chinese_xlnet_mid_pytorch')
# 加载微调后的模型
trained_mdl = torch.load('../tmp_models/new/xlnetclsfy_epoch5loss:0.12982.pt')
print(f'完成！')

print(f'加载测试集...')
with open('../all_file_list_test', 'rb') as f:
    test_list = pickle.load(f)
print('完成！')

got_the_numpy = False

# i = 0

for single_doc in tqdm(test_list):
    single_doc = single_doc.replace(' ', '')
    single_doc = single_doc[0:2048]
    
    inputs = xl_tokenizer(single_doc, return_tensors='pt')
    outputs = trained_mdl(**inputs)
    last_hidden_state = outputs.hidden_states[-1]

    # 取分类token的embedding
    cls_emb = last_hidden_state[0][-1]
    cls_emb = cls_emb.detach().cpu().numpy()

    if not got_the_numpy:
        all_emb_np = cls_emb
        got_the_numpy = True
    else:
        all_emb_np = np.vstack((all_emb_np, cls_emb))

    # print(all_emb_np.shape)
    # i += 1
    # if i == 10:
    #     break

# print(all_emb_np.shape, type(all_emb_np))

# 将全部test数据集每篇文档的feature_emb表示序列化到磁盘
# print(f'将 emb ndarray 结果序列化到磁盘...')
# np.save('./feature_emb_test.npy', all_emb_np)
# print(f'完成！')

