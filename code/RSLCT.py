import random
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax
from tqdm import tqdm

import numpy as np
import pylab as plt
import time

from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score as ari, contingency_matrix
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import kmc2

np.random.seed(0)


def transform_dataset(data, labels, iter):
    start_time = time.time()
    data = StandardScaler().fit_transform(data)

    search_size = 5

    mus = np.zeros(shape=(data.shape[0], data.shape[1]))
    sigmas = np.zeros(shape=(data.shape[0], 1))
    generative = np.zeros(shape=(data.shape[0], data.shape[1]))
    X = data

    kdt = KDTree(data)
    dist, idx = kdt.query(X, k=search_size + 1)
    for i in tqdm(range(iter)):
        # show the trend of transforming
        if i % 250 == 0:
            plt.figure()
            plt.title(f"iteration {i}")
            if X.shape[1] > 2:
                X_tsne = TSNE(n_components=2).fit_transform(X)
                plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=8)
            else:
                plt.scatter(X[:, 0], X[:, 1], c=labels, s=8)
            plt.show()
            plt.close()
            # kdt = KDTree(X)
            # dist, idx = kdt.query(X, k=search_size+1)

        mus = np.mean(X[idx[:, 1:]], axis=1)
        sigmas = np.min(dist[:, 1:], axis=1)

        for n in range(X.shape[0]):
            # generate new points
            generative[n] = np.random.normal(mus[n], sigmas[n])

        X = StandardScaler().fit_transform(generative)
    cost_time = time.time() - start_time

    return X, cost_time


class RSLC(object):
    """
    Double Sampling Likelihood Clustering
    """
    def __init__(self, params={}, clustering_algo=None):
        # initialize parameters
        self.params = {
            "k_min": 2,
            "k_max": 30,
            "near_num": 5,
            "proportion": 0.7,
            "max_iter_times": 20,
            "iter_avg": 1,
            "distributed": False,
            "algorithm": "kmeans"
        }

        self.params.update(params)
        self.losses = []
        self.k_values = [i for i in range(self.params["k_min"], self.params["k_max"])]
        self.clustering_algo = clustering_algo

        self.pred_centers = []
        self.pred_labels = []

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

        return s1, s2

    def loss_func(self, X, centers, nearest_clusters_idx, sigma2, **kwargs):
        summary = 0.

        # Calculate the loss function
        dist_loss = 0.
        for k in range(centers.shape[0]):
            cluster_data = X[nearest_clusters_idx == k]
            if cluster_data.shape[0] < 1:
                continue
            distance = euclidean_distances(cluster_data, centers[k].reshape(1, -1))
            dist_loss += np.sum(distance) / (2 * np.sqrt(cluster_data.shape[0]))
        summary += dist_loss

        summary += np.sqrt(centers.shape[0])
        summary += centers.shape[0] / np.sqrt(X.shape[0])

        return summary

    def predict_k(self, X):
        print("================== Start Predict ====================")
        self.s1, self.s2 = self.double_sample(X)

        s1 = self.s1
        s2 = self.s2
        N = s1.shape[0]
        D = s1.shape[1]
        print(f"data (n,d) = ({N}, {D})")
        k_min = self.params["k_min"]
        k_max = self.params["k_max"]
        k_range = range(k_min, k_max)

        losses = []

        print(f"search cluster_number from {k_min} to {k_max - 1}")
        cost_time = 0.
        for cluster_number in tqdm(k_range):
            near_num = min(cluster_number, self.params["near_num"])

            # clustering more times
            avg_loss = 0.
            for _ in range(self.params["iter_avg"]):
                # =================== Clustering ===================
                new_centers = None
                pred_label = None
                start_time = time.time()
                # kmeans form
                if "kmeans" == self.params["algorithm"]:
                    kmeans = KMeans(n_clusters=cluster_number, max_iter=self.params["max_iter_times"]).fit(s1)
                    new_centers = kmeans.cluster_centers_
                    pred_label = kmeans.labels_
                    km_iter = kmeans.n_iter_
                else:
                    # =================== Var GMM ======================
                    init_centers = kmc2.kmc2(s1, cluster_number, afkmc2=True)
                    sigma = 1.
                    new_centers = init_centers

                    nearest_idx = (np.asarray([np.random.choice(cluster_number, near_num) for _ in range(N)])
                                   .astype(np.int32))

                    # init posterior
                    G_c = np.asarray([
                        np.concatenate(
                            [np.asarray([c]),
                             np.random.permutation(
                                 np.delete(np.arange(cluster_number), np.asarray([c])))
                             ], axis=0)[:near_num]
                        for c in range(cluster_number)])
                    for _ in range(self.params["max_iter_times"]):
                        G_n = [np.unique(np.concatenate(G_c[near_cluster_of_data])) for near_cluster_of_data in
                               nearest_idx]

                        posterior = (-np.inf) * np.ones((N, cluster_number), dtype=np.float64)
                        # calculate p(x_n, mu_c) based on the nearest clusters
                        for n, (c, x) in enumerate(zip(G_n, s1)):
                            # distance = np.square(dist[n])
                            distance = np.square(np.linalg.norm(x - new_centers[c], axis=-1))
                            posterior[n, c] = (-1. / (2. * sigma) * distance)
                        # the index of the nearest points
                        nearest_idx = np.argpartition(posterior, cluster_number - near_num, axis=1)[:, -near_num:]

                        idx = np.zeros((cluster_number, N), dtype=bool)
                        rows = np.arange(N).T
                        idx.T[rows[:, np.newaxis], nearest_idx] = True

                        p_trunc = -np.inf * np.ones_like(posterior, dtype=np.float64)
                        for c in range(cluster_number):
                            # idx[c] = (1, n) 如果idx[c, n]为True，则代表点n是与簇c相近的
                            p_trunc[idx[c], c] = posterior[idx[c], c]

                        # ======= Update G_c =================================
                        # (1, n) cluster_datapoints[n] 代表离点n最近的簇
                        cluster_datapoints = np.argmax(posterior, axis=1)
                        G_c = np.empty((cluster_number, near_num), dtype=np.int32)
                        for c in range(cluster_number):
                            cluster_distances = posterior[cluster_datapoints == c, :]
                            # 选择无穷值的mask用来筛选 filter the infinite mask
                            mask = ~np.isfinite(cluster_distances)
                            # 根据mask结果隐藏无穷值 mask the infinite
                            masked_cluster_distances = np.ma.array(cluster_distances, mask=mask)
                            # print(f"mask cluster distance [not mean] shape = {masked_cluster_distances.shape}")
                            # 计算该簇的点对各簇的平均概率（忽视mask掉的无穷值） calculate the average probability
                            mean_cluster_distance = masked_cluster_distances.mean(axis=0)
                            # 将mask的值用负无穷填充 fill with negative infinite
                            mean_cluster_distance = np.ma.filled(mean_cluster_distance, -np.inf)
                            # 该簇的点对该簇的似然值最大为0【其他为负】 max likelihood
                            mean_cluster_distance[c] = 0.
                            # 计算该簇各点的各近簇的似然值，即G_c[c]=(1, 400)代表最属于该簇的点属于其他簇的概率 likelihood probability
                            G_c[c] = np.argpartition(mean_cluster_distance, cluster_number - near_num)[-near_num:]

                        posterior = softmax(p_trunc, axis=1)

                        # ======= Update cluster center ======================
                        sum_posterior = np.sum(posterior, axis=0)
                        new_centers = np.tensordot(posterior, s1, axes=(0, 0))
                        new_centers[sum_posterior != 0] /= sum_posterior[sum_posterior != 0, np.newaxis]

                        # ======= Update data sigma ==========================
                        my_sum_X2 = np.zeros(cluster_number, dtype=np.float64)
                        for r, x in zip(posterior, s1):
                            X2_term = r * np.inner(x, x)
                            # my_sum_X2, sum_error = SCS(my_sum_X2, X2_term + sum_error)
                            my_sum_X2 += X2_term
                        sum_X2 = my_sum_X2
                        centers2 = np.asarray([np.inner(mean, mean) for mean in new_centers])
                        sigma = np.sum(sum_X2 - centers2 * sum_posterior) / float(N * D)
                    kdt = KDTree(new_centers)
                    dist, idx = kdt.query(s1)
                    pred_label = idx[:, 0]
                cost_time += time.time() - start_time

                avg_loss += self.loss_func(s1, new_centers, pred_label, None)
                self.pred_centers.append(new_centers)
                # self.pred_labels.append(pred_label)
            losses.append(avg_loss / self.params["max_iter_times"])
        self.training_time = cost_time
        # ======== numpy array ==============
        losses = np.array(losses)

        self.losses = losses

        self.pred_k = k_min + np.argmin(self.losses)
        print("================== End Predict ====================")
        return self.pred_k

    def plot_points(self, data):
        if data.shape[1] > 2 or self.pred_k == -1:
            return
        # kmeans = KMeans(n_clusters=self.pred_k).fit(data)
        # centers = kmeans.cluster_centers_
        # labels = kmeans.labels_

        if self.pred_centers is None or len(self.pred_centers) == 0:
            return
        centers = self.pred_centers[self.pred_k - self.params["k_min"]]
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
            plt.scatter(k_true, self.losses[k_true - self.params["k_min"]])
        plt.show()
        plt.close()

    def metrics(self, data, labels_true):
        if self.pred_k == -1:
            print("do not have predict k value")
            return None, None, None
        # kmeans = KMeans(n_clusters=self.pred_k).fit(data)
        # labels_pred = kmeans.labels_

        mu = self.pred_centers[self.pred_k - self.params["k_min"]]
        kdt = KDTree(mu)
        dist, idx = kdt.query(data)
        labels_pred = idx[:, 0]

        ari_score = ari(labels_true, labels_pred)
        nmi_score = nmi(labels_true, labels_pred)
        ami_score = ami(labels_true, labels_pred)
        purity_score = self.purity(labels_true, labels_pred)
        return ari_score, ami_score, nmi_score, purity_score

    def purity(self, y_true, y_pred):
        matrix = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)


if __name__ == '__main__':
    dataset = "flame"
    data = np.load(f"../datasets/{dataset}/{dataset}_data.npy")
    labels = np.load(f"../datasets/{dataset}/{dataset}_labels.npy")
    true_k = np.unique(labels).shape[0]

    params = {
        "k_min": 2,
        "k_max": true_k + 10,  # the searching range limitation
        "near_num": 5,  # the nearest search num for Var-GMM
        "proportion": 0.7,  # the sampling proportion
        "max_iter_times": 200,  # the max iteration times for k-means
        "iter_avg": 1,  # the iteration times
        "algorithm": "kmeans",  # the used algorithm
        "is_transformed": True,  # RSLC or RSLCT
        "transform_times": 1000
    }
    gen_time = 0
    if params['is_transformed']:
        data, gen_time = transform_dataset(data, labels, params['transform_times'])

        plt.figure()
        plt.title(f"transformed dataset {dataset}")
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.show()
        plt.close()

    model = RSLC(params)
    pred_k = model.predict_k(data)

    model.plot_losses(true_k)
    pred_labels = model.pred_labels
    cost_time = model.training_time
    ari_score, ami_score, nmi_score, purity_score = model.metrics(data, labels)

    record = f"================"\
             f"\ndata shape={data.shape}"\
             f"\ncluster number={true_k}"\
             f"\npredict cluster number={pred_k}"\
             f"\ncost time={cost_time} + {gen_time} = {cost_time + gen_time}"\
             f"\nari={ari_score}, ami={ami_score}, nmi={nmi_score}, purity={purity_score}"\
             f"\nparameters={model.params}"\
             f"\n================"
    # with open(f"test_dataset_{dataset}.txt", "a") as f:
    #     f.write(record)

    print(record)


