import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets import make_blobs

n = 50000
d = 100
k = 200
X, y = make_blobs(n_samples=n, n_features=d, centers=k, cluster_std=0.1)
print("X:{}".format(X.shape))
print("labels:{}".format(y.shape))
print("cluster:{}".format(np.unique(y).shape[0]))
# plt.figure(figsize=(8, 8))
# plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.savefig("../generated/Gaussian_{}_{}.png".format(n, k))
# plt.show()

np.save(f"../generated/self-comparison/comparison_n{n}_d{d}_k{k}_data.npy", X)
np.save(f"../generated/self-comparison/comparison_n{n}_d{d}_k{k}_labels.npy", y)

# np.save("../generated/Gaussian_{}_k_{}_value".format(n, k), X)
# np.save("../generated/Gaussian_{}_k_{}_label".format(n, k), y)

# z = []
# for i in range(n):
#     x = list(X[i])
#     x.append(y[i])
#     z.append(x)
# z = np.array(z)
# print("z:{}".format(z.shape))
# np.save("../generated/Gaussian_{}_k_{}".format(n, k), z)

# centers = []
# cluster_std = [500 for _ in range(1000)]
#
# for i in range(1000):
#     centers.append(
#         [(i % 50 - 25) * 1000, (i / 50 - 10) * 1000]
#     )
#
# x, label = make_blobs(n_samples=[10 for _ in range(1000)], cluster_std=cluster_std, centers=centers, n_features=2,
#                       random_state=1)
#
# np.save("Gaussian{}_value.npy".format(1000), x)
# np.save("Gaussian{}_label.npy".format(1000), label)
#
# plt.scatter(x[:, 0], x[:, 1], s=10)
# plt.show()
#
#
# def gen_clusters():
#     xx = np.array([np.random.randn(2) * 100 for _ in range(100)])
#     data = xx
#     for i in range(999):
#         new_cluster = np.array([np.random.randn(2) * 100 for _ in range(100)])
#         new_cluster -= np.array([[(i - 500) % 40 * 100, (i - 500) / 40 * 100] for _ in range(100)])
#         data = np.append(data, new_cluster, 0)
#
#     return np.round(data, 4)
#
#
# def init_cluster(mean, cov, num):
#     return np.random.multivariate_normal(mean, cov, num)
#
#
# def append_new_cluster(mean, cov, num, data):
#     gen = np.random.multivariate_normal(mean, cov, num)
#     data = np.append(data, gen, 0)
#     return data
#
#
# def save_data(data, filename):
#     with open(filename, 'w') as file:
#         for i in range(data.shape[0]):
#             file.write(str(data[i, 0]) + ',' + str(data[i, 1]) + '\n')
#
#
# def load_data(filename):
#     data = []
#     with open(filename, 'r') as file:
#         for line in file.readlines():
#             data.append([float(i) for i in line.split(',')])
#     return np.array(data)
#
#
# def show_scatter(data):
#     x, y = data.T
#     plt.scatter(x, y)
#     plt.axis()
#     plt.title("scatter")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.show()
#
# # data = gen_clusters()
# # save_data(data, '3clusters.txt')
# # d = load_data('3clusters.txt')
# # show_scatter(d)
