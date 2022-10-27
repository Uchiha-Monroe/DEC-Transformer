import numpy as np
from collections import Counter


# 分别加载标签和预测的numpy数组
labels = np.load('./cluster_label.npy')
kmeans_final = np.load('./cluster_final_train.npy')

all_correct_nums = 0
for cls_index in range(20):
    tmp_np = kmeans_final[labels == cls_index]    
    ground_truth = Counter(tmp_np).most_common(1)[0][0]
    ground_truth_nums = Counter(tmp_np).most_common(1)[0][1]
    all_correct_nums += ground_truth_nums

print(f'准确率是{all_correct_nums / len(labels)}')



