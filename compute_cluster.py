import numpy as np
import numpy as np
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AgglomerativeClustering, Birch
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import time

import logging

from torch import threshold
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载预料的tf-idf结果矩阵
logger.info('start reading tf-idf matrix...')
tik = time.time()
tf_idf_matrix = np.load('./fudan_tf-idf.npy')
tok = time.time()
logger.info(f'successfully read tf-idf matrix,time costed {tok - tik}s.')

# SVD矩阵分解
logger.info(f'正在进行稀疏矩阵转换...')
sparse_tf_idf_matrix = sparse.csr_matrix(tf_idf_matrix)
logger.info(f'稀疏矩阵转换完成')
# svd = TruncatedSVD(n_components=500, n_iter=20)
# logger.info(f'正在进行SVD分解...')
# tik = time.time()
# lsa_matrix = svd.fit_transform(sparse_tf_idf_matrix)
# print(lsa_matrix.shape)
# tok = time.time()
# logger.info(f'SVD矩阵分解完成，耗时{tok - tik} s。')

# 使用KMeans算法进行聚类运算
logger.info('start clustering...')
tik = time.time()
kmeans_final = KMeans(n_clusters=20).fit(sparse_tf_idf_matrix)
tok = time.time()
logger.info(f'succseefully cluster, time costed {tok - tik}s.')

# 将结果序列化到磁盘
np.save('./cluster_final_train', kmeans_final.labels_)

