import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE



# 加载fine_tune方法的预测结果
fine_tuned_result = torch.load('./fine_tune/fine_tuned_result.pt')
fine_tuned_result = fine_tuned_result.numpy()

feature_emb_test = np.load('./feature_ensemble/feature_emb_test.npy')

tsne = TSNE(n_components=2)
tsne = tsne.fit_transform(feature_emb_test)

fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1 = sns.scatterplot(tsne[:, 0], tsne[:, 1], c=fine_tuned_result, marker='+', cmap='tab20')
fig1.savefig("temp2.jpg", bbox_inches=None)