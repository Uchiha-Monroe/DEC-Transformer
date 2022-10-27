import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification

from neg_data import NormalDataset


import logging
logging.basicConfig(level=logging.INFO)
my_logger = logging.getLogger(__name__)

if __name__ == '__main__':

    batch_size = 4
    # 加载数据
    my_logger.info(f'加载数据和标签...')
    all_file_list_dir = '../all_file_list'
    all_file_dict_dir = '../all_file_dict'
    cluster_label_dir = '../cluster_label.npy'
    train_dataset = NormalDataset(all_file_list_dir, cluster_label_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    my_logger.info(f'数据加载成功')
    # print(len(train_dataloader))
    

    # 模型路径
    cn_bert_wwm_model_path = '/home/oem/mydisk/outer_models/cn_models/chinese-bert_chinese_wwm_pytorch'
    cn_xlnet_path = '/home/oem/mydisk/outer_models/cn_models/chinese_xlnet_mid_pytorch'

    # 使用bert_wwwm模型
    # cn_bert_tokenizer = BertTokenizer.from_pretrained(cn_bert_wwm_model_path)
    # cn_bert_model = BertForSequenceClassification.from_pretrained(cn_bert_wwm_model_path)

    # 使用xlnet模型
    my_logger.info(f'加载XLNet...')
    cn_xlnet_tokenizer = XLNetTokenizer.from_pretrained(cn_xlnet_path)
    cn_xlnet_clsfy_model = XLNetForSequenceClassification.from_pretrained(cn_xlnet_path)
    my_logger.info(f'加载成功')

    
        
    if torch.cuda.is_available():
        cn_xlnet_clsfy_model.cuda()
        my_logger.info(f'cuda有效，模型上cuda成功')

    if torch.cuda.device_count() > 1:
        my_logger.info(f'可使用 {torch.cuda.device_count()} 个GPU')
        cn_xlnet_clsfy_model = nn.DataParallel(cn_xlnet_clsfy_model)
        my_logger.info(f'模型成功部署在多个GPU上')

    optimizer = Adam(cn_xlnet_clsfy_model.parameters(), lr=5e-6)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.2)

    # 模型保存路径
    model_saved_path = '/home/oem/mydisk/cluster/Fudan/tmp_models'
    train_epochs = 5
    total_loss = 0

    for i_epoch in range(train_epochs):
        
        my_logger.info(f'starting epoch {i_epoch + 1}')
        
        with tqdm(total=len(train_dataloader), desc=f'Epoch {i_epoch + 1}') as pbar:         
            for index, data in enumerate(train_dataloader):
                # doc_i是元组，label_i是shape为[4]的tensor
                doc_i, labels = data
                labels.cuda()
                
                # print(type(doc_i))
                # print(len(doc_i))
                # print(label_i)
                # break

                # 对输入数据的文本长度进行限制并重组
                corpus = []
                for single_doc in doc_i:
                    single_doc = single_doc.replace(' ', '')
                    corpus.append(single_doc[:2048])
                
                inputs = cn_xlnet_tokenizer(corpus, return_tensors='pt', padding=True, )
                outputs = cn_xlnet_clsfy_model(**inputs, labels=labels)
                
                # 维度调试
                # print(outputs.logits.shape)
                

                # 误差计算及参数更新
                optimizer.zero_grad()
                cal_loss = outputs.loss.mean()
                cal_loss.backward()
                optimizer.step()

                # 累积误差更新
                total_loss += outputs.loss.sum().item()
                total_step = (i_epoch * len(train_dataloader) + index + 1) * batch_size
                mean_loss = total_loss / total_step

                pbar.set_description('loss:{0:1.5f}'.format(mean_loss))
                pbar.update(1)
                # print(outputs.loss)
                # if index == 10:
                #     break

                # 每个epoch保存一个模型
                if index == len(train_dataloader) - 1:
                    saved_path = model_saved_path + '/new/' + 'xlnetclsfy_' + 'epoch' + str(i_epoch + 1) + 'loss:' + str(round(mean_loss, 5)) + '.pt'
                    my_logger.info(f'正在保存第{i_epoch + 1}个epoch的模型到{saved_path}...')
                    torch.save(cn_xlnet_clsfy_model, saved_path)
                    my_logger.info(f'模型保存成功')
            
            scheduler.step()




