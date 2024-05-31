import time

import pylab as plt
import numpy as np
import torch
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler

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
    return variance
    # return 6.0


def get_nearest_clusters_idx(X, centers, near_num):
    distance_matrix = euclidean_distances(X, centers)
    # 排序后的索引
    Zn = distance_matrix.argsort()
    # 取排序后各行的前near_num个
    Zn = Zn[:, :near_num]
    return Zn


def iter_time_logger(current_k, k_max, last_time):
    current_time = time.time()
    print(f"searching k={current_k} "
          f"and may need for {np.round((current_time - last_time) / 10 * (k_max - current_k), 2)}s")
    return current_time


def update_centers(X, centers, nearest_clusters_idx):
    mu = np.zeros(centers.shape)
    sigma2 = get_sigma2(X, centers, nearest_clusters_idx[:, 0])
    s = get_s(X, centers, sigma2)
    # print(s)
    for c in range(centers.shape[0]):
        count = 0
        for n in range(X.shape[0]):
            if nearest_clusters_idx[n, 0] == c:
                mu[c] += s[n, c] * X[n]
                count += 1
        if count > 0:
            mu[c] /= count
    return mu


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
            "near_num": 3,
            "proportion": 0.7,
            "iter_times": 1
        }

        self.params.update(params)
        self.losses = []
        self.k_values = [i for i in range(self.params["k_min"], self.params["k_max"])]
        self.clustering_algo = clustering_algo

        self.res_centers = None
        self.res_labels = None

        self.s1 = None
        self.s2 = None
        self.pred_k = -1

        self.training_time = None

    def double_sample(self, X):
        """
        double sampling algorithm

        Parameters:
            X: dataset to sample
            proportion: proportion of double sample block size

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
        co = - 1. / X.shape[0]
        s = get_s(X, centers, sigma2)

        summary = 0.
        # ================== point to cluster =========================
        distance = euclidean_distances(X, centers, squared=True)
        for n in range(X.shape[0]):
            # summary += np.sum(- s[n, nearest_clusters_idx[n, 0]] * distance[n, nearest_clusters_idx[n, 0]] / (2 * sigma2))
            summary += np.sum(- distance[n, nearest_clusters_idx[n, 0]] / 2 / sigma2)
            # summary += np.mean(- distance[n, :] / 2 / sigma2)
            # summary += np.sum(- s[n, nearest_clusters_idx[n, 0]])
            # print(np.sum(- distance[n, nearest_clusters_idx[n, 0]] / 2 / sigma2))
            # print(np.sum(- s[n, nearest_clusters_idx[n, 0]]))

            # # GMM
            # exp_arg = -1. / (2. * sigma2) * np.square(np.linalg.norm(X[n] - centers, axis=1))
            # shift = np.max(exp_arg) - 707. + np.log(centers.shape[0])
            # summary += np.log(np.sum(np.exp(exp_arg - shif t))) + shift
        summary *= co
        # print(f"point to cluster loss:{summary}")

        # ================== between clusters =======================
        # cluster_between_distance = euclidean_distances(centers, centers)
        # # summary += np.log(centers.shape[0]) * np.min(cluster_between_distance)
        # # print(np.argsort(cluster_between_distance, axis=1)[:, 1])
        # min_cluster_idx = np.argsort(cluster_between_distance, axis=1)[:, 1]
        # dist = 0.
        # for i in range(centers.shape[0]):
        #     dist += cluster_between_distance[i, min_cluster_idx[i]]
        # summary -= dist / centers.shape[0]

        # for c in range(centers.shape[0]):
        #     cluster = X[nearest_clusters_idx[:, 0] == c]
        #     summary += np.sum(euclidean_distances(cluster, centers[c].reshape(1, -1)))

        # if "old_labels" in kwargs:
        #     new_label = KMeans(n_clusters=centers.shape[0]).fit(X).labels_
        #     ari_loss = - ari(new_label, kwargs["old_labels"])
        #     summary += ari_loss
        #     # print(f"ari loss: {ari_loss}")

        # ================== cluster number penalty ===================
        summary += np.log(centers.shape[0])
        # print(f"cluster number penalty:{np.log(centers.shape[0])}")
        # ================== dimension adjust =======================
        summary += X.shape[1] / 2. * np.log(2 * np.pi * sigma2)
        # print(f"dimension penalty loss: {X.shape[1] / 2. * np.log(2 * np.pi * sigma2)}")
        summary += centers.shape[0] / X.shape[0]
        # print(f"data shape loss:{centers.shape[0] / X.shape[0]}")
        return summary

    def predict_k(self, X):
        print("================== Start Predict ====================")
        start_time = time.time()
        s1, s2 = self.double_sample(X)

        self.s1 = s1
        self.s2 = s2
        k_min = self.params["k_min"]
        k_max = self.params["k_max"]

        min_loss = np.inf
        losses = []
        losses1 = []
        losses2 = []
        print(f"search k from {k_min} to {k_max}")
        for k in range(k_min, k_max):
            near_num = min(k, self.params["near_num"])

            if (k - k_min) % 10 == 0 and k > k_min:
                start_time = iter_time_logger(k, k_max, start_time)

            # cluster_centers1 = kmc2.kmc2(s1, k)
            # cluster_centers2 = kmc2.kmc2(s2, k)
            # cluster_centers1 = kmeans_pytorch.initialize(s1, k)

            # # ========= Training ===========
            # iter_times = self.params["iter_times"]
            # for i in range(iter_times):
            #     nearest_clusters_idx1 = get_nearest_clusters_idx(s1, cluster_centers1, near_num)
            #     cluster_centers1 = update_centers(s1, cluster_centers1, nearest_clusters_idx1)
            #
            #     nearest_clusters_idx2 = get_nearest_clusters_idx(s2, cluster_centers2, near_num)
            #     cluster_centers2 = update_centers(s2, cluster_centers2, nearest_clusters_idx2)
            #
            # nearest_clusters_idx1 = get_nearest_clusters_idx(s1, cluster_centers1, near_num)
            # sigma1 = get_sigma2(s1, cluster_centers1, nearest_clusters_idx1[:, 0])
            # nearest_clusters_idx2 = get_nearest_clusters_idx(s2, cluster_centers2, near_num)
            # sigma2 = get_sigma2(s2, cluster_centers2, nearest_clusters_idx2[:, 0])

            # clustering more time
            avg_loss = 0
            for i in range(self.params["iter_times"]):
                # =================== Clustering ===================
                kmeans = KMeans(n_clusters=k).fit(s1)
                cluster_centers1 = kmeans.cluster_centers_
                labels1 = kmeans.labels_
                # plt.figure()
                # plt.scatter(s1[:, 0], s1[:, 1], c=labels1, s=5)
                # plt.scatter(cluster_centers1[:, 0], cluster_centers1[:, 1])
                # plt.show()
                # plt.close()

                kmeans = KMeans(n_clusters=k).fit(s2)
                cluster_centers2 = kmeans.cluster_centers_
                labels2 = kmeans.labels_

                # kmeans = KMeans(n_clusters=k).fit(X)
                # cluster_centers = kmeans.cluster_centers_
                # labels = kmeans.labels_

                nearest_clusters_idx1 = get_nearest_clusters_idx(s1, cluster_centers1, near_num)
                sigma1 = get_sigma2(s1, cluster_centers1, labels1)
                # print(f"k={k}, sigma={sigma1}")

                nearest_clusters_idx2 = get_nearest_clusters_idx(s2, cluster_centers2, near_num)
                sigma2 = get_sigma2(s2, cluster_centers2, labels2)

                # nearest_clusters_idx = get_nearest_clusters_idx(X, cluster_centers, near_num)
                # sigma = get_sigma2(X, cluster_centers, labels)

                # print(f"\n==================== cluster: {k} ===================")
                loss1 = self.loss_func(s1, cluster_centers1, nearest_clusters_idx1, sigma1)
                loss2 = self.loss_func(s2, cluster_centers2, nearest_clusters_idx2, sigma2)
                # losses.append((loss1+loss2)/2)
                # loss = self.loss_func(X, cluster_centers, nearest_clusters_idx, sigma)

                # # gap statistic
                # s1_disp = np.sum(np.min(euclidean_distances(s1, cluster_centers1), axis=1)) / s1.shape[0]
                # s2_disp = np.sum(np.min(euclidean_distances(s2, cluster_centers2), axis=1)) / s2.shape[0]
                # loss1 += - (np.log(s1_disp) - np.log(s2_disp))
                # print(f"gap statistic loss: {- (np.log(s1_disp) - np.log(s2_disp))}")

                avg_loss += (loss1 + loss2) / 2
            # print(f"cluster {k} : {avg_loss / self.params['iter_times']}")
            losses1.append(avg_loss / self.params["iter_times"])
            # losses2.append(loss2)
            # losses.append(loss)

            # if losses[-1] < min_loss:
            #     self.res_centers = cluster_centers
            #     self.res_labels = cluster_labels
            #     self.pred_k = k
            #     min_loss = losses[-1]

        # ======== numpy array ==============
        losses1 = np.array(losses1)
        # losses2 = np.array(losses2)
        # losses = np.array(losses)

        self.training_time = time.time() - start_time
        self.losses = losses1
        # self.pred_k = np.argmin(losses) + k_min

        # # ============= test losses ===================
        # plt.figure()
        # plt.title("test losses on samples")
        # plt.plot(range(k_min, k_max), losses1, label="loss1")
        # # plt.plot(range(k_min, k_max), losses2, label="loss2")
        # # plt.plot(range(k_min, k_max), losses, label="loss full")
        # # plt.plot(range(k_min, k_max), losses1 - np.min(losses1) + losses2 - np.min(losses2),
        # #          label="loss sum 1+2")
        # plt.legend()
        # plt.show()
        # plt.close()

        # print(f"loss1 k={k_min + np.argmin(losses1)}")
        # print(f"loss2 k={k_min + np.argmin(losses2)}")
        # print(f"loss full k={k_min + np.argmin(losses)}")
        # print(f'loss sum 1+2 k={k_min + np.argmin(losses1 - np.min(losses1) + losses2 - np.min(losses2))}')
        # print(f"loss sum 1+2+0 k={k_min + np.argmin(losses1 + losses2 + losses)}")
        self.pred_k = k_min + np.argmin(self.losses)
        print("================== End Predict ====================")
        return self.pred_k

    def plot_points(self, data):
        # if self.res_centers is None:
        #     print("Not fit yet")
        #     return
        # if self.s1.shape[1] > 2:
        #     print("High dimension to plot")
        #     return
        # plt.figure()
        # plt.scatter(self.s1[:, 0], self.s1[:, 1], c=self.res_labels[:])
        # plt.scatter(self.res_centers[:, 0], self.res_centers[:, 1], c="red")
        # plt.show()
        # plt.close()

        if data.shape[1] > 2 or self.pred_k == -1:
            return
        kmeans = KMeans(n_clusters=self.pred_k).fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
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
        kmeans = KMeans(n_clusters=self.pred_k).fit(data)
        labels_pred = kmeans.labels_

        ari_score = ari(labels_true, labels_pred)
        nmi_score = nmi(labels_true, labels_pred)
        purity_score = purity(labels_true, labels_pred)
        return ari_score, nmi_score, purity_score


def run_test():
    params = {
        "k_min": 2,
        "k_max": 41
    }
    model = DSLC(params)
    dataset = "r15"

    data = np.load(f"../datasets/{dataset}/{dataset}_data.npy")
    # data = StandardScaler().fit_transform(data)
    labels = np.load(f"../datasets/{dataset}/{dataset}_labels.npy")

    print(f"true k={np.unique(labels).shape[0]}")

    k = model.predict_k(data)
    model.plot_losses()
    # model.plot_points(data)
    cost_time = model.training_time
    ari_score, nmi_score, purity_score = model.metrics(data, labels)
    print(f"ari={ari_score}, nmi={nmi_score}, purity={purity_score}")

    record = (f"\ndataset {dataset}, shape:{data.shape}"
              f"\ncost for {cost_time}s={cost_time / 60}min={cost_time / 3600}h"
              f"\nk_true={np.unique(labels).shape[0]}, predict k={k}"
              f"\nari={ari_score}, nmi={nmi_score}, purity={purity_score}"
              f"\n")
    print(record)

    # with open("dslc_record.txt", "a") as f:
    #     f.write(record)


if __name__ == '__main__':
    run_test()
