# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d
import numpy as np

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始

    # 1.对所有样本进行正则化
    data_mean = np.mean(data, axis=0)
    data_normalized = data - data_mean

    # 2.计算样本协方差矩阵
    #covmat = np.cov(data_normalized, rowvar=0)
    H = np.dot(data_normalized.T, data_normalized)

    # 3.对协方差矩阵做特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(H)

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors, data_mean


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    #point_cloud_pynt = PyntCloud.from_file("/Users/renqian/Downloads/program/cloud_data/11.ply")
    #point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    pc_txt_path = "/Users/hangz/PycharmProjects/pc/guitar_0060.txt"
    raw_data = np.genfromtxt(pc_txt_path, delimiter=",")
    pc = raw_data[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    #o3d.visualization.draw_geometries([pcd])    # 显示原始点云



    # 从点云中获取点，只对点进行处理
    #points = point_cloud_pynt.points
    #print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v, origin = PCA(pc)
    point_cloud_vector = v[:, 2] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', v[:, 0])
    print('the second orientation of this pointcloud is: ', v[:, 1])
    # TODO: 此处只显示了点云，还没有显示PCA
    point = [origin, v[:, 0], v[:, 1]]
    lines = [[0, 1], [0, 2]]
    colors = [[1, 0, 0], [0, 1, 0]]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd, line_set])
    
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = []
    # 作业2
    # 屏蔽开始

    for i in range(pc.shape[0]):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 10)
        k_nearest_point = np.asarray(pcd.points)[idx, :]
        nw, nv, n_origin = PCA(k_nearest_point)
        normals.append(nv[:, 2])
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # 屏蔽结束

    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
