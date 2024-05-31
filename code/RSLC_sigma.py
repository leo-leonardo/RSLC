import datetime
import time

import matplotlib
import numpy
import pylab as plt
import numpy as np
import torch
from numpy import float64
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# matplotlib.use('module://matplotlib_inline.backend_inline')
matplotlib.use('TKAgg')

import kmc2


def purity(y_true, y_pred):
    matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)


def get_sigma2(X, centers, pred_labels):
    variances = []
    for c in range(centers.shape[0]):
        cluster_points = X[pred_labels == c]
        variances.append(np.var(cluster_points))
    variance = np.min(variances)
    return variance * 2.
    # return (6.236491003460207 + 4.27898009215501) / 2


def get_nearest_clusters_idx(X, centers, near_num):
    distance_matrix = euclidean_distances(X, centers)
    # 排序后的索引
    Zn = distance_matrix.argsort()
    # 取排序后各行的前near_num个
    Zn = Zn[:, :near_num]
    return Zn


def log_iter_time(current_k, k_max, last_time):
    current_time = time.time()
    print(f"searching k={current_k} "
          f"and may still need for {np.round((current_time - last_time) / 10 * (k_max - current_k), 2)}s")
    return current_time


# def update_centers(X, centers, nearest_clusters_idx):
#     mu = np.zeros(centers.shape)
#     sigma2 = get_sigma2(X, centers, nearest_clusters_idx[:, 0])
#     s = get_s(X, centers, sigma2)
#     # print(s)
#     for c in range(centers.shape[0]):
#         count = 0
#         for n in range(X.shape[0]):
#             if nearest_clusters_idx[n, 0] == c:
#                 mu[c] += s[n, c] * X[n]
#                 count += 1
#         if count > 0:
#             mu[c] /= count
#     return mu


def get_s(X, centers, sigma2):
    s = np.zeros(shape=(X.shape[0], centers.shape[0]), dtype=np.float32)
    distances = euclidean_distances(X, centers, squared=True)
    for n in range(X.shape[0]):
        exps = np.exp(- distances[n, :] / (2 * sigma2))
        s[n] = exps / np.sum(exps)
        # s[n] = softmax(exps)
        # s[n] = exps
    return s


class DSLC(object):
    """
    Double Sampling Likelihood Clustering

    """

    def __init__(self, params={}, clustering_algo=None):
        self.params = {
            "k_min": 2,
            "k_max": 30,
            "near_num": 5,
            "proportion": 0.7,
            "iter_times": 1
        }

        self.params.update(params)
        self.losses = []
        self.k_values = [i for i in range(self.params["k_min"], self.params["k_max"])]
        self.clustering_algo = clustering_algo

        self.res_centers = []
        self.res_labels = []

        self.s1 = None
        self.s2 = None
        self.pred_k = -1
        self.sigma2 = 0

        self.training_time = None

    def double_sample(self, X):
        """
        double sampling algorithm

        Parameters:
            X: dataset to sample
            self.params['proportion']: proportion of double sample block size

        Return:
            s1: sample block 1
            s2: sample block 2
        """
        # Default to sample half of dataset
        if self.params["proportion"] is None:
            self.params["proportion"] = 0.5
        n = X.shape[0]
        permutation = np.random.permutation(n)
        s1 = X[permutation[:int(self.params["proportion"] * n)]]
        s2 = X[permutation[int(self.params["proportion"] * n):]]

        # # print(s1)
        # epsmax = np.sqrt(X.shape[1])
        # perturbation_kwargs = [eps for eps in np.linspace(0.0, epsmax, num=10)]
        # for perturbation in perturbation_kwargs:
        #     s1 += np.random.uniform(low=-perturbation, high=perturbation, size=s1.shape)
        # # print(s1)

        return s1, s2

    def loss_func(self, X, centers, nearest_clusters_idx, sigma2, **kwargs):
        summary = 0.

        co = 1. / X.shape[0]
        # s = get_s(X, centers, sigma2)

        # ================== point to cluster =========================
        # distance = euclidean_distances(X, centers, squared=True)
        # dist_loss = 0.
        # for n in range(X.shape[0]):
        #     # summary += np.sum(s[n, nearest_clusters_idx[n, :]] * distance[n, nearest_clusters_idx[n, :]] / 2 / sigma2)
        #     dist_loss += np.sum(distance[n, nearest_clusters_idx[n]] / 2 / sigma2)
        # summary += co * dist_loss

        # Testing
        dist_loss = 0.
        for k in range(centers.shape[0]):
            cluster_data = X[nearest_clusters_idx == k]
            if cluster_data.shape[0] == 0:
                continue
            distance = euclidean_distances(cluster_data, centers[k].reshape(1, -1))
            # dist_loss += np.sum(distance[nearest_clusters_idx == k, k] / (2 * np.sqrt(cluster_data.shape[0])))
            dist_loss += np.sum(distance) / (2 * np.sqrt(cluster_data.shape[0]))
        summary += dist_loss
        # print(f"point to cluster loss:{summary}")

        # ================== cluster number penalty ===================
        # summary += np.log(centers.shape[0])
        summary += np.sqrt(centers.shape[0])
        # print(f"cluster number penalty:{np.log(centers.shape[0])}")
        # ================== dimension adjust =======================
        # summary += X.shape[1] / 2. * np.log(2 * np.pi * sigma2)
        # print(f"dimension penalty loss: {X.shape[1] / 2. * np.log(2 * np.pi * sigma2)}")
        summary += centers.shape[0] / np.sqrt(X.shape[0])
        # print(f"data shape loss:{centers.shape[0] / X.shape[0]}")

        # # BIC
        # summary += X.shape[0] / 2. * np.log(2 * np.pi)
        # summary += X.shape[0] * X.shape[1] / 2. * np.log(sigma2)
        # summary += - np.log((X.shape[0] - centers.shape[0]) / 2.)
        # summary += - X.shape[0] * np.log(X.shape[0])
        # summary += X.shape[0] * np.log(X.shape[0] / self.params["proportion"])

        return summary

    def predict_k(self, X):
        print("================== Start Predict ====================")
        start_time = time.time()
        self.s1, self.s2 = self.double_sample(X)

        s1 = self.s1
        s2 = self.s2
        N = s1.shape[0]
        D = s1.shape[1]
        print(f"data (n,d) = ({N}, {D})")
        k_min = self.params["k_min"]
        k_max = self.params["k_max"]
        k_range = range(k_min, k_max)

        min_loss = np.inf
        losses = []
        losses1 = []
        losses2 = []

        print(f"search cluster_number from {k_min} to {k_max - 1}")
        for cluster_number in tqdm(k_range):
            # print(f"cluster_number={cluster_number}")
            near_num = min(cluster_number, self.params["near_num"])

            # if (cluster_number - k_min) % 10 == 0 and cluster_number > k_min:
            #     start_time = log_iter_time(cluster_number, k_max, start_time)

            # cluster_centers1 = kmc2.kmc2(s1, cluster_number)
            # cluster_centers2 = kmc2.kmc2(s2, cluster_number)
            # cluster_centers1 = kmeans_pytorch.initialize(s1, cluster_number)

            params = {
                'Niter': 40,
                # origin GMM
                'algorithm': 'var-GMM-S',
                'C': cluster_number,
                'Cprime': 5,
                'G': 5,
                'Ninit': 10,
                'dataset': 'BIRCH',
                'VERBOSE': {
                    'll': False,
                    'fe': False,
                    'qe': False,
                    'cs': False,
                    'nd': True,
                    'np': np.inf,
                }
            }

            # clustering more time
            avg_loss = 0.
            for _ in range(self.params["iter_times"]):
                params.update({'C': cluster_number})
                # gmm = GMM(params)
                # gmm.fit(s1)
                # new_centers = gmm.means

                # =================== Clustering ===================
                kmeans = KMeans(n_clusters=cluster_number).fit(s1)
                new_centers = kmeans.cluster_centers_
                pred_label = kmeans.labels_
                km_iter = kmeans.n_iter_

                # # =================== Var GMM ======================
                # init_centers = kmc2.kmc2(s1, cluster_number, afkmc2=True)
                # sigma = 1.
                # new_centers = init_centers
                #
                # nearest_idx = (np.asarray([np.random.choice(cluster_number, near_num) for _ in range(N)])
                #                .astype(np.int32))
                #
                # # init posterior
                # G_c = np.asarray([
                #     np.concatenate(
                #         [np.asarray([c]),
                #          np.random.permutation(
                #              np.delete(np.arange(cluster_number), np.asarray([c])))
                #          ], axis=0)[:near_num]
                #     for c in range(cluster_number)])
                # # print(f"G_c={G_c}")
                # # print(f"G_c.shape={G_c.shape}")
                # for _ in range(params["Niter"]):
                #     """
                #     var gmm
                #
                #     resp = softmax(-1/(2 * sigma) * d(x,mean))
                #
                #     means = np.tensordot(resp, X, axes=(0, 0))
                #     means[resp != 0] /= resp[resp != 0, np.newaxis]
                #
                #     init sigma = 1
                #     sigma = resp * np.inner(x, x) - resp * np.inner(mean, mean)
                #     sigma /= N * D
                #     """
                #
                #     G_n = [np.unique(np.concatenate(G_c[near_cluster_of_data])) for near_cluster_of_data in nearest_idx]
                #
                #     posterior = (-np.inf) * np.ones((N, cluster_number), dtype=np.float64)
                #     # calculate p(x_n, mu_c) based on the nearest clusters
                #     for n, (c, x) in enumerate(zip(G_n, s1)):
                #         # distance = np.square(dist[n])
                #         distance = np.square(np.linalg.norm(x - new_centers[c], axis=-1))
                #         # 大小一致
                #         # print("square distance =", np.square(dist[n])
                #         # print("square norm =", np.square(np.linalg.norm(x - new_centers[c], axis=-1)))
                #         posterior[n, c] = (-1. / (2. * sigma) * distance)
                #     nearest_idx = np.argpartition(posterior, cluster_number - near_num, axis=1)[:, -near_num:]
                #
                #     idx = np.zeros((cluster_number, N), dtype=bool)
                #     rows = np.arange(N).T
                #     idx.T[rows[:, np.newaxis], nearest_idx] = True
                #
                #     p_trunc = -np.inf * np.ones_like(posterior, dtype=np.float64)
                #     for c in range(cluster_number):
                #         # idx[c] = (1, n) 如果idx[c, n]为True，则代表点n是与簇c相近的
                #         p_trunc[idx[c], c] = posterior[idx[c], c]
                #
                #     # ======= Update G_c =================================
                #     # (1, n) cluster_datapoints[n] 代表离点n最近的簇
                #     cluster_datapoints = np.argmax(posterior, axis=1)
                #     G_c = np.empty((cluster_number, near_num), dtype=np.int32)
                #     for c in range(cluster_number):
                #         cluster_distances = posterior[cluster_datapoints == c, :]
                #         # print(f"cluster_distances = {cluster_distances}")
                #
                #         # all_cluster_distances = np.concatenate(comm.allgather(cluster_distances))
                #         # the following is the buffered (more stable) version of
                #         # the above allgather command:
                #         # 选择无穷值的mask用来筛选
                #         mask = ~np.isfinite(cluster_distances)
                #         # 根据mask结果隐藏无穷值
                #         masked_cluster_distances = np.ma.array(cluster_distances, mask=mask)
                #         # print(f"mask cluster distance [not mean] shape = {masked_cluster_distances.shape}")
                #         # 计算该簇的点对各簇的平均概率（忽视mask掉的无穷值）
                #         mean_cluster_distance = masked_cluster_distances.mean(axis=0)
                #         # 将mask的值用负无穷填充
                #         mean_cluster_distance = np.ma.filled(mean_cluster_distance, -np.inf)
                #         # 该簇的点对该簇的似然值最大为0【其他为负】
                #         mean_cluster_distance[c] = 0.
                #         # 计算该簇各点的各近簇的似然值，即G_c[c]=(1, 400)代表最属于该簇的点属于其他簇的概率
                #         G_c[c] = np.argpartition(mean_cluster_distance, cluster_number - near_num)[-near_num:]
                #
                #     posterior = softmax(p_trunc, axis=1)
                #
                #     # ======= Update cluster center ======================
                #     sum_posterior = np.sum(posterior, axis=0)
                #     new_centers = np.tensordot(posterior, s1, axes=(0, 0))
                #     new_centers[sum_posterior != 0] /= sum_posterior[sum_posterior != 0, np.newaxis]
                #
                #     # ======= Update data sigma ==========================
                #     my_sum_X2 = np.zeros(cluster_number, dtype=np.float64)
                #     for r, x in zip(posterior, s1):
                #         X2_term = r * np.inner(x, x)
                #         # my_sum_X2, sum_error = SCS(my_sum_X2, X2_term + sum_error)
                #         my_sum_X2 += X2_term
                #     sum_X2 = my_sum_X2
                #     centers2 = np.asarray([np.inner(mean, mean) for mean in new_centers])
                #     sigma = np.sum(sum_X2 - centers2 * sum_posterior) / float(N * D)
                # kdt = KDTree(new_centers)
                # dist, idx = kdt.query(s1)
                # pred_label = idx[:, 0]

                # if cluster_number == 100:
                #     plt.figure()
                #     plt.title(f"new_centers {cluster_number}")
                #     plt.scatter(s1[:, 0], s1[:, 1], s=5)
                #     plt.scatter(new_centers[:, 0], new_centers[:, 1], s=10, c="black")
                #     plt.show()
                #     # plt.savefig("test_gmm.png")
                #     plt.close()

                # nearest_clusters_idx1 = get_nearest_clusters_idx(s1, cluster_centers1, near_num)
                # sigma1 = get_sigma2(s2, cluster_centers2, labels2)
                # sigma1 = self.sigma2
                # print(f"cluster_number={cluster_number}, sigma={sigma1}")

                # nearest_clusters_idx2 = get_nearest_clusters_idx(s2, cluster_centers2, near_num)
                # sigma2 = self.sigma2

                # nearest_clusters_idx = get_nearest_clusters_idx(X, cluster_centers, near_num)
                # sigma = get_sigma2(X, cluster_centers, labels)

                # print(f"\n==================== cluster: {cluster_number} ===================")
                # sigma1 = None
                # loss1 = self.loss_func(s1, cluster_centers1, labels1, sigma1)
                # loss2 = self.loss_func(s2, cluster_centers2, nearest_clusters_idx2, sigma2)
                # losses.append((loss1+loss2)/2)
                # loss = self.loss_func(X, cluster_centers, nearest_clusters_idx, sigma)

                # avg_loss += (loss1 + loss2) / 2
                # avg_loss += loss1
                # avg_loss += self.loss_func(s1, new_centers, nearest_idx[:, 0], None)
                avg_loss += self.loss_func(s1, new_centers, pred_label, None)
                self.res_centers.append(new_centers)
                self.res_labels.append(pred_label)
            # print(f"cluster {cluster_number} : {avg_loss / self.params['iter_times']}")
            losses1.append(avg_loss / self.params["iter_times"])
            # losses2.append(loss2)
            # losses.append(loss)

            # if losses[-1] < min_loss:
            #     self.res_centers = cluster_centers
            #     self.res_labels = cluster_labels
            #     self.pred_k = cluster_number
            #     min_loss = losses[-1]

        # ======== numpy array ==============
        losses1 = np.array(losses1)
        # losses2 = np.array(losses2)
        # losses = np.array(losses)

        self.training_time = time.time() - start_time
        self.losses = losses1

        self.pred_k = k_min + np.argmin(self.losses)
        print("================== End Predict ====================")
        return self.pred_k

    def plot_points(self, data):
        if data.shape[1] > 2 or self.pred_k == -1:
            return
        # kmeans = KMeans(n_clusters=self.pred_k).fit(data)
        # centers = kmeans.cluster_centers_
        # labels = kmeans.labels_

        centers = self.res_centers[self.pred_k - self.params["k_min"]]
        kdt = KDTree(centers)
        dist, labels = kdt.query(data)
        labels = labels[:, 0]

        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=5, c=labels[:])
        plt.scatter(centers[:, 0], centers[:, 1], s=10, c="red")
        plt.show()
        plt.close()

    def plot_losses(self, k_true=None):
        plt.figure()
        plt.xlabel("k-clusters")
        plt.ylabel("losses")
        plt.plot(self.k_values, self.losses)
        if k_true is not None:
            plt.scatter(k_true, self.losses[k_true])
        plt.show()
        plt.close()

    def metrics(self, data, labels_true):
        if self.pred_k == -1:
            print("do not have predict k value")
            return None, None, None
        # kmeans = KMeans(n_clusters=self.pred_k).fit(data)
        # labels_pred = kmeans.labels_

        mu = self.res_centers[self.pred_k - self.params["k_min"]]
        kdt = KDTree(mu)
        dist, idx = kdt.query(data)
        labels_pred = idx[:, 0]

        ari_score = ari(labels_true, labels_pred)
        nmi_score = nmi(labels_true, labels_pred)
        purity_score = purity(labels_true, labels_pred)
        return ari_score, nmi_score, purity_score


def run_test():
    params = {
        "k_min": 1,
        "k_max": 30,
        "proportion": 0.7
    }
    model = DSLC(params)

    dataset = "r15"
    data = np.load(f"../datasets/{dataset}/{dataset}_data.npy")
    data = StandardScaler().fit_transform(data)
    labels = np.load(f"../datasets/{dataset}/{dataset}_labels.npy")

    print(f"true k={np.unique(labels).shape[0]}")

    k = model.predict_k(data)
    model.plot_losses()
    model.plot_points(data)
    cost_time = model.training_time
    ari_score, nmi_score, purity_score = model.metrics(data, labels)
    print(f"ari={ari_score}, nmi={nmi_score}, purity={purity_score}")

    record = (f"\ndataset {dataset}, shape:{data.shape}"
              f"\ncost for {cost_time}s={cost_time / 60}min={cost_time / 3600}h"
              f"\nk_true={np.unique(labels).shape[0]}, predict k={k}"
              f"\nari={ari_score}, nmi={nmi_score}, purity={purity_score}"
              f"\n")

    # with open("dslc_record.txt", "a") as f:
    #     f.write(record)

    print(record)


if __name__ == '__main__':
    run_test()
