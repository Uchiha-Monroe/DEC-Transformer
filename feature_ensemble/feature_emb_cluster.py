import numpy as np
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AgglomerativeClustering, Birch
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import time
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import pandas as pd

# 加载结果矩阵
print(f'加载feature_emb结果矩阵...')
feature_emb_test = np.load('./feature_emb_test.npy')
print(f'完成')

# 聚类操作
print('开始聚类...')
kmeans = KMeans(n_clusters=20).fit(feature_emb_test)
print('完成')

np.save('./feature_emb_cluster.npy', kmeans.labels_)

# 可视化
label_pred = kmeans.labels_.reshape(-1, 1)
centroids = kmeans.cluster_centers_

tsne = TSNE(n_components=2)
tsne = tsne.fit_transform(feature_emb_test)

plt.scatter(tsne[:, 0], tsne[:, 1], c=label_pred, cmap='tab20')
plt.show()