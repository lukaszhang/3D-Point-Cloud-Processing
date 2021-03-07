# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct

import octree as octree
import kdtree as kdtree
from scipy import spatial
from result_set import KNNResultSet, RadiusNNResultSet

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

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    #root_dir = '/Users/hangz/PycharmProjects/pythonProject/data' # 数据集路径
    #cat = os.listdir(root_dir)
    #iteration_num = len(cat)

    # 只处理一个bin文件
    filename = '/Users/hangz/PycharmProjects/pythonProject/000000.bin'
    db_np = read_velodyne_bin(filename)
    print(db_np.shape)

    print("octree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0

    # 统计构建Octree的时间
    begin_t = time.time()
    root = octree.octree_construction(db_np, leaf_size, min_extent)
    construction_time_sum += time.time() - begin_t

    # Octree knn search
    begin_t = time.time()
    for i in range(len(db_np)):
    #for i in range(30000):
        result_set = KNNResultSet(capacity=k)
        query = db_np[i, :]
        octree.octree_knn_search(root, db_np, result_set, query)
    knn_time_sum += time.time() - begin_t

    # Octree radius search
    #begin_t = time.time()
    #for i in range(len(db_np)):
    #for i in range(30000):
    #    query = db_np[i, :]
    #    result_set = RadiusNNResultSet(radius=radius)
    #    octree.octree_radius_search_fast(root, db_np, result_set, query)
    #radius_time_sum += time.time() - begin_t

    # brute search time
    #begin_t = time.time()
    #for i in range(len(db_np)):
    #for i in range(30000):
    #    query = db_np[i, :]
    #    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    #    nn_idx = np.argsort(diff)
    #    nn_dist = diff[nn_idx]
    #brute_time_sum += time.time() - begin_t
    print("Octree: build %.3fms, knn %.3fms, radius %.3fms, brute %.3fms" % (construction_time_sum*1000,
                                                                             knn_time_sum*1000,
                                                                             radius_time_sum*1000,
                                                                             brute_time_sum*1000))

    print("kdtree --------------")
    # spatial.KDTree
    construction_time_sum = 0
    knn_time_sum = 0
    # construction
    begin_t = time.time()
    tree = spatial.KDTree(db_np, leaf_size)
    construction_time_sum += time.time() - begin_t
    # search
    begin_t = time.time()
    #tree.query(x=db_np[0:30000, :], k=8)
    tree.query(x=db_np, k=8)
    knn_time_sum += time.time() - begin_t
    print("Kdtree_spatial: build %.3fms, knn %.3fms" % (construction_time_sum * 1000, knn_time_sum * 1000))


    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0

    begin_t = time.time()
    root = kdtree.kdtree_construction(db_np, leaf_size)
    construction_time_sum += time.time() - begin_t

    # KDtree knn
    begin_t = time.time()
    for i in range(len(db_np)):
    #for i in range(30000):
        result_set = KNNResultSet(capacity=k)
        query = db_np[i, :]
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
    knn_time_sum += time.time() - begin_t

    # KDtree radius search
    #begin_t = time.time()
    #for i in range(len(db_np)):
    #for i in range(30000):
    #    query = db_np[i, :]
    #    result_set = RadiusNNResultSet(radius=radius)
    #    kdtree.kdtree_radius_search(root, db_np, result_set, query)
    #radius_time_sum += time.time() - begin_t

    # KDtree brute search
    #begin_t = time.time()
    #for i in range(len(db_np)):
    #for i in range(30000):
    #    query = db_np[i, :]
    #    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    #    nn_idx = np.argsort(diff)
    #    nn_dist = diff[nn_idx]
    #brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3fms, knn %.3fms, radius %.3fms, brute %.3fms" % (construction_time_sum * 1000,
                                                                             knn_time_sum * 1000,
                                                                             radius_time_sum * 1000,
                                                                             brute_time_sum * 1000))



if __name__ == '__main__':
    main()