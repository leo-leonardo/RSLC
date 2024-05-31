import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax
from tqdm import tqdm
import random

random.seed(1)

dataset_name = "r15"
data = np.load(f"../datasets/{dataset_name}/{dataset_name}_data.npy")
labels = np.load(f"../datasets/{dataset_name}/{dataset_name}_labels.npy")
data = StandardScaler().fit_transform(data)

search_size = 5

# 点点距离
# distance = pairwise_distances(data, data)
# 距离排序nn_data_idx[i,j]对应i点最近的第j点
# nn_data_idx = np.argsort(distance, axis=1)[:, 1:search_size+1]


# # 点点距离
# distance = pairwise_distances(data, data)
# # 距离排序nn_data_idx[i,j]对应i点最近的第j点
# nn_data_idx = np.argsort(distance, axis=1)[:, 1:search_size+1]

mus = np.zeros(shape=(data.shape[0], data.shape[1]))
sigmas = np.zeros(shape=(data.shape[0], 1))
generative = np.zeros(shape=(data.shape[0], data.shape[1]))
X = data

start_time = time.time()
print("build kdTree")
kdt = KDTree(data)
dist, idx = kdt.query(X, k=search_size+1)
iter = 2000
losses = []
print("Start searching")
for i in tqdm(range(iter)):
    # 阶段性展示
    # if i % 250 == 0:
    #     plt.figure()
    #     plt.title(f"iteration {i}")
    #     if X.shape[1] > 2:
    #         X_tsne = TSNE(n_components=2).fit_transform(X)
    #         plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=8)
    #     else:
    #         plt.scatter(X[:, 0], X[:, 1], c=labels, s=8)
    #     plt.show()
    #     plt.close()
    #     # 每次重新使用KDTree更新位置信息，都会使点更锁定
    #     # kdt = KDTree(X)
    #     # dist, idx = kdt.query(X, k=search_size+1)

    # 查找各个点最近的search_size个点
    # 查找到的索引为三维，对应i,j,k=第i点的第j近点的第k维度，因此针对axis=1展开，每次取列平均，即j个的平均
    # posterior = softmax(- dist[:, 1:] + np.max(dist[:], axis=1, keepdims=True))
    posterior = softmax(- dist[:, 1:])
    # print(posterior)

    # 加权坐标作为新坐标
    # mus = np.mean(X[idx[:, 1:]], axis=1)
    mus = np.einsum('ijk,ij->ik', X[idx[:, 1:]], posterior)
    # 最近点距离作为方差扩散
    sigmas = np.min(dist[:, 1:], axis=1)
    # sigmas = np.log(dist[:, 1])

    for n in range(X.shape[0]):
        # 生成新点
        generative[n] = np.random.normal(mus[n], sigmas[n])

    loss = 0.
    for n in range(X.shape[0]):
        diff = X[n] - generative[n]
        # print(diff)
        loss += -np.log(np.sum(np.square(diff)))
    # print(loss / X.shape[0])
    losses.append(loss / X.shape[0])

    # for n in range(X.shape[0]):
        # print(data[nn_data_idx[n]])
        # direction_diff = X[nn_data_idx[n]] - X[n]
        # direction = np.mean(direction_diff, axis=0)
        # dist, idx = kdt.query([X[n]], k=search_size+1)
        # mus[n] = np.mean(X[nn_data_idx[n]], axis=0)
        # mus[n] += np.mean(np.var(X[nn_data_idx[n]], axis=0))
        # mus[n] += direction
        # mus[n] += direction / np.linalg.norm(direction)
        # print(np.var(data[nn_data_idx[n]], axis=0))
        # sigmas[n] = np.mean(np.var(X[nn_data_idx[n]], axis=0))
        # sigmas[n] = np.mean(distance[n, nn_data_idx[n]], axis=0)
        # generative[n] = np.random.normal(mus[n], sigmas[n])

    # 持续标准化数据
    X = StandardScaler().fit_transform(generative)
cost_time = time.time() - start_time

# print(losses)
# plt.figure()
# plt.title("loss in generative")
# plt.plot(range(iter), np.array(losses))
# plt.show()
# plt.close()

with open(f"generate_time.txt", "a") as f:
    f.write("============"
            f"\nTransform {dataset_name} for {iter} times"
            f"\nCost time={cost_time}"
            f"\n==================="
            f"\n")

# np.save(f"dataset/{dataset_name}/{dataset_name}_embedding.npy", X)


# plt.figure()
# plt.title(f"{dataset_name} search {search_size} points and {iter} times for generating")
# # plt.scatter(data[:, 0], data[:, 1], c="red", s=5)
# # plt.scatter(mus[:, 0], mus[:, 1], c="black", s=3)
# # plt.scatter(generative[:, 0], generative[:, 1], c=labels, s=8)
# # plt.show()
# # plt.close()
#
# kmeans = KMeans(n_clusters=np.unique(labels).shape[0]).fit(X)
# labels_pred = kmeans.labels_
# centers = kmeans.cluster_centers_
#
# if X.shape[1] > 2:
#     X_tsne = TSNE(n_components=2).fit_transform(X)
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=8)
# else:
#     plt.scatter(X[:, 0], X[:, 1], c=labels, s=8)
# # plt.scatter(centers[:, 0], centers[:, 1], c="black", s=10)
# plt.show()
# plt.close()
#
# ari = adjusted_rand_score(labels, labels_pred)
# print(ari)
#
# origin_kmeans = KMeans(n_clusters=np.unique(labels).shape[0]).fit(data)
# labels_pred = origin_kmeans.labels_
# ari = adjusted_rand_score(labels, labels_pred)
# print(ari)

