import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

# 加载TF-IDF LSA KMeans的预测结果
computed_labels = np.load('cluster_label_test.npy')

feature_emb_test = np.load('./feature_ensemble/feature_emb_test.npy')

tsne = TSNE(n_components=2)
tsne = tsne.fit_transform(feature_emb_test)


fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2 = sns.scatterplot(tsne[:, 0], tsne[:, 1], c=computed_labels, marker='+', cmap='tab20')
fig2.savefig("temp2.eps", bbox_inches=None, pad_inches=0.2)

plt.show()