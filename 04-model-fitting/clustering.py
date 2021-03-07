# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
import random
import math
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import time

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmented_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    idx_segmented = []
    segmented_cloud = []
    tau = 0.2
    best_model = np.zeros((1, 4))
    pre_total = 0
    P = 0.99
    n = len(data)
    outlier_ratio = 0.4
    iters = math.ceil(math.log(1 - P) / math.log(1 - pow(0.6, 3)))
    i = 0
    while i < iters:
        #ground_cloud = []
        idx_ground = []
        # choose three points randomly
        sample_index = np.random.choice(n, 3, replace=False)
        point1 = data[sample_index[0]]
        point2 = data[sample_index[1]]
        point3 = data[sample_index[2]]

        # solve the normal vector
        point1_2 = point2 - point1
        point1_3 = point3 - point1
        N = np.cross(point1_2, point1_3)
        a = N[0]
        b = N[1]
        c = N[2]
        d = -np.dot(point1, N)

        # calculate the distance between all points and the plane
        distance = abs(np.dot(data, N.reshape((3, 1))) + np.ones((n, 1))*d) / np.linalg.norm(N)
        idx_ground = (distance <= tau)
        total_inlier = np.sum(idx_ground == True)
        i += 1
        # if the number of points are larger than previous, change the best model parameter
        if total_inlier > pre_total:
            pre_total = total_inlier
            best_model = np.array([a, b, c, d])
            #best_distance = distance

        # if enough points are cooresponding with the plane, break
        if total_inlier > 1.2 * n * (1 - outlier_ratio):
            best_model = np.array([a, b, c, d])
            #best_distance = distance
            break
    print("iters = %f" %iters)
    seg_distance = np.abs(np.dot(data, best_model[0:3].reshape(3, 1)) + np.ones((n, 1)) * best_model[3]) / np.linalg.norm(best_model[0:3])
    #idx_segmented = np.logical_not(idx_ground)
    #ground_cloud = data[idx_ground]
    #segmented_cloud = data[idx_segmented]
    segmented_ground = data[np.where(seg_distance < tau)[0]]
    segmented_cloud = data[np.where(seg_distance > tau)[0]]

    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmented_cloud.shape[0])
    return segmented_ground, segmented_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    N, _ = data.shape
    iter_num = 0

    # index=0: unvisited; index=1: visited
    visited_point = np.zeros((N, 1))
    # label=-1: unlabeled; label=0: noise; label=others: different labels
    cluster_label = -np.ones((N, 1), dtype=int)
    label = 0

    min_samples = 28
    neigh = NearestNeighbors(radius=0.82, algorithm='kd_tree')
    neigh.fit(data)

    while (visited_point == 0).any():
        # randomly choose an unvisited point as seed
        unvisited_indices = np.where(visited_point == 0)[0]
        print("processing: ==========", 100 - np.floor(unvisited_indices.shape[0] / N * 100), "%==========")
        seed_indix = np.random.choice(unvisited_indices, 1)
        seed = data[seed_indix]
        visited_point[seed_indix] = 1

        rng = neigh.radius_neighbors(seed)
        neigh_indices = np.unique(np.asarray(rng[1][0]))

        if neigh_indices.shape[0] > min_samples:
            label += 1
            cluster_label[seed_indix] = label
            cluster_num = neigh_indices.shape[0]
            i = 0
            while i < cluster_num:
                point_idx = neigh_indices[i]
                if visited_point[point_idx] == 0:
                    visited_point[point_idx] = 1

                if cluster_label[point_idx] != -1:
                    i += 1
                    continue
                else:
                    cluster_label[point_idx] = label

                point = data[point_idx].reshape((1, 3))
                rng = neigh.radius_neighbors(point)
                new_indices = np.asarray(rng[1][0])

                if new_indices.shape[0] > min_samples:
                    neigh_indices = np.unique(np.hstack((neigh_indices, new_indices)))

                i += 1
                cluster_num = neigh_indices.shape[0]

        else:
            cluster_label[seed_indix] = 0


    # 屏蔽结束
    #cluster_label = cluster_label.flatten()
    clusters_label = cluster_label.reshape((N, 1))
    print(cluster_label)
    return cluster_label


def plot_clusters_o3d(segmented_ground, segmented_cloud, cluster_index):
    def colormap(c):
        color_cycle = [[1, 0, 0],  # red
                       [1, 0.5, 0],  # orange
                       [1, 1, 0],  # yellow
                       [0, 1, 0],  # green
                       [0, 1, 1],  # sky
                       [0, 0, 1],  # blue
                       [0.1, 0.1, 0.48],  # navi
                       [1, 0, 1]]  # purple
        # outlier:
        if c == 0:
            color = [0] * 3
        # surrouding object:
        else:
            color = color_cycle[np.int(c % 8)]

        return color

    # ground element:
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(segmented_ground)
    pcd_ground.colors = o3d.utility.Vector3dVector(
        [
            [0.28]*3 for i in range(segmented_ground.shape[0])
        ]
    )

    # surrounding object elements:
    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(segmented_cloud)
    # num_clusters = max(cluster_index) + 1
    pcd_objects.colors = o3d.utility.Vector3dVector(
        [
            colormap(c) for c in cluster_index
        ]
    )
    # visualize:
    # o3d.visualization.draw_geometries([pcd_ground, pcd_objects])
    o3d.visualization.draw_geometries([pcd_ground, pcd_objects])
# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    #ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color='b')
    plt.show()

def main():
    root_dir = '/Users/hangz/Desktop/kitti_point_clouds' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[0:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        start = time.time()
        segmented_ground, segmented_points = ground_segmentation(data=origin_points)
        print("Segmentation time is ", time.time() - start, "s")
        start = time.time()
        cluster_index = clustering(segmented_points)
        print("clustering time is ", time.time() - start, "s")

        #plot_clusters(segmented_points, cluster_index)
        plot_clusters_o3d(segmented_ground, segmented_points, cluster_index=cluster_index)

if __name__ == '__main__':
    main()
