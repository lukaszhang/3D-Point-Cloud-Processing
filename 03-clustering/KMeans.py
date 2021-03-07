# 文件功能： 实现 K-Means 算法

import numpy as np
import random
import math
import time

class K_Means(object):
    center = np.empty((0, 1))
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        points_num = data.shape[0]
        center_indices = np.random.choice(points_num, self.k_, replace=False)
        K_Means.center = np.zeros((self.k_, data.shape[1]))

        for i in range(self.k_):
            K_Means.center[i] = data[center_indices[i]]

        error = 1e10
        iter_num = 0
        pre_center = K_Means.center
        label_count = np.zeros((self.k_, 1), dtype=np.int_)
        while error > self.tolerance_ and iter_num < self.max_iter_:
            # sort all points to each clusters
            label = K_Means.predict(self, data)

            # recompute the center of each clusters
            center_sum = np.zeros((self.k_, 2), dtype=np.float32)
            for i in range(points_num):
                center_sum[label[i]] += data[i]
                for j in range(self.k_):
                    if label[i] == j:
                        label_count[j] += 1

            for i in range(self.k_):
                K_Means.center[i] = center_sum[i]/label_count[i]

            # compute error
            error = 0
            for i in range(self.k_):
                error += np.linalg.norm(pre_center[i]-K_Means.center[i])

            iter_num += 1
            pre_center = K_Means.center

        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        points_num = p_datas.shape[0]
        label = np.zeros((points_num), dtype=np.int_)
        for i in range(points_num):
            min_diff = 1e10
            for j in range(self.k_):
                diff = np.linalg.norm(p_datas[i]-K_Means.center[j])
                if diff < min_diff:
                    label[i] = j
                    min_diff = diff

        result = label
        # 屏蔽结束
        return result


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

