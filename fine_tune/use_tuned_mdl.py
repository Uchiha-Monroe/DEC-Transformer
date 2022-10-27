# -*- coding: utf-8 -*-

from transformers import XLNetForSequenceClassification, XLNetTokenizer
from transformers import XLNetConfig
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import pickle

import logging
logging.basicConfig(level=logging.INFO)
my_logger = logging.getLogger(__name__)

if __name__ == '__main__':

    cn_xlnet_path = '/home/oem/mydisk/outer_models/cn_models/chinese_xlnet_mid_pytorch'

    # xlnet_configuration = XLNetConfig('/home/oem/mydisk/outer_models/cn_models/chinese_xlnet_mid_pytorch/config.json')
    # xlnet_clsfy_model = XLNetForSequenceClassification(xlnet_configuration)
    # xlnet_clsfy_model = nn.DataParallel(xlnet_clsfy_model)


    # 初始化模型
    my_logger.info('初始化模型...')
    xlnet_tokenizer = XLNetTokenizer.from_pretrained('/home/oem/mydisk/outer_models/cn_models/chinese_xlnet_mid_pytorch')
    xlnet_clsfy_model = torch.load('../tmp_models/new/xlnetclsfy_epoch5loss:0.12982.pt')


    
    # 加载预处理后的文本数据
    with open('all_file_list_test', 'rb') as f:
        all_file_list = pickle.load(f)

    result = []
    # i = 0

    for single_doc in tqdm(all_file_list):
        single_doc = single_doc.replace(' ', '')
        single_doc = single_doc[:2048]
        inputs = xlnet_tokenizer(single_doc, return_tensors='pt')
        outputs = xlnet_clsfy_model(**inputs)
        # print(outputs[0], outputs[0].shape)
        result.append(outputs[0].argmax().item())
        # i += 1
        # if i == 100:
        #     print(result)

    tuned_result = torch.tensor(result)

    # 储存计算结果
    my_logger.info('将fine_tuned后的模型预测结果写入磁盘...')
    result_path = './fine_tuned_result.pt'
    torch.save(tuned_result, result_path)
    my_logger.info('写入完成')

    # 计算结果
    labels_np = np.load('../cluster_label.npy')
    labels = torch.tensor(labels_np)

    ratio = (tuned_result == labels).to(dtype=int).sum().item() / len(labels)
    my_logger.info(f'准确率是：{round(ratio * 100, 2)}%')
        

    

