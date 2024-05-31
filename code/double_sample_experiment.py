import numpy as np
import pylab as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def plot_samples_origin(data, labels, dataset_name):
    plt.figure()
    plt.title(dataset_name)
    plt.subplot(411)
    plt.scatter(data[:, 0], data[:, 1], s=5, c=labels)

    def subplot_proportion(x, y, pos, data, labels, permutation, proportion):
        n = data.shape[0]
        s1 = data[permutation[:int(proportion * n)]]
        s2 = data[permutation[int(proportion * n):]]
        labels1 = labels[permutation[:int(proportion * n)]]
        labels2 = labels[permutation[int(proportion * n):]]

        plt.subplot(x, y, pos)
        plt.scatter(s1[:, 0], s1[:, 1], s=5, c=labels1)
        plt.subplot(x, y, pos + 1)
        plt.scatter(s2[:, 0], s2[:, 1], s=5, c=labels2)

    n = data.shape[0]
    permutation = np.random.permutation(n)
    subplot_proportion(4, 2, 3, data, labels, permutation, 0.1)
    subplot_proportion(4, 2, 5, data, labels, permutation, 0.3)
    subplot_proportion(4, 2, 7, data, labels, permutation, 0.5)

    plt.show()
    plt.close()


def sub_sample_kmeans(data, labels, dataset_name):

    n = data.shape[0]
    permutation = np.random.permutation(n)

    sub_sample_1 = data[permutation[:int(n * 0.1)]]
    sub_sample_3 = data[permutation[:int(n * 0.3)]]
    sub_sample_5 = data[permutation[:int(n * 0.5)]]
    sub_sample_7 = data[permutation[:int(n * 0.7)]]
    sub_sample_9 = data[permutation[:int(n * 0.9)]]

    def plot_sample(data, proportion):
        plt.figure()
        plt.title(f"{proportion} dataset")
        plt.scatter(data[:, 0], data[:, 1])
        plt.show()
        plt.close()

    plot_sample(sub_sample_1, 0.1)
    plot_sample(sub_sample_3, 0.3)
    plot_sample(sub_sample_5, 0.5)
    plot_sample(sub_sample_7, 0.7)
    plot_sample(sub_sample_9, 0.9)

# np.random.seed(0)

dataset_name = "compound"
data_path = f"../datasets/{dataset_name}/{dataset_name}_data.npy"
label_path = f"../datasets/{dataset_name}/{dataset_name}_labels.npy"

data = np.load(data_path)
data = StandardScaler().fit_transform(data)
labels = np.load(label_path)

sub_sample_kmeans(data, labels, dataset_name)

# data = TSNE(n_components=2).fit_transform(data)
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.show()
plt.close()
