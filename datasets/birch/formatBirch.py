# # if __name__ == '__main__':
# #     str_row = ""
# #     with open("birch1.txt", "r+", encoding="UTF8") as f:
# #         for row in f:
# #             point = row.strip().split()
# #             str_row = str_row + point[0] + "\t" + point[1] + "\n"
# #
# #     with open("birch1_format.txt", "w", encoding="UTF8") as f:
# #         f.write(str_row)

import numpy as np

# data = np.loadtxt("birch2_split.txt", delimiter=",")
# print(f"data={data}")
# np.save("birch2_data.npy", data)
#
# labels = np.loadtxt("birch2_labels.txt")
# labels = np.array(labels, dtype=int)
# np.save("birch2_labels.npy", labels)
#
# # from sklearn.cluster import KMeans
# # km = KMeans(n_clusters=100)
# # labels = km.fit(data).labels_
# # labels = np.array(labels, dtype=int)
# # print(f"labels={labels}")
# # np.save("birch_labels.npy", labels)
#
# import pylab as plt
#
# plt.figure()
# plt.scatter(data[:, 0], data[:, 1], c=labels, s=1)
# plt.savefig("birch1_km.jpg")
# plt.show()

centers = np.loadtxt("birch_center.txt")
print(centers)
np.save("birch_centers.npy", centers)

