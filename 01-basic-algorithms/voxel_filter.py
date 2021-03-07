# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d
import numpy as np
import random

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, if_mean=False):
    filtered_points = []
    # 作业3
    # 屏蔽开始

    # 计算边界点
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)

    # 计算voxel grid维度
    Dx = (x_max - x_min)//leaf_size + 1
    Dy = (y_max - y_min)//leaf_size + 1
    Dz = (z_max - z_min)//leaf_size + 1
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))

    # 计算每个点的Voxel索引
    h = list()
    for i in range(len(point_cloud)):
        hx = (point_cloud[i][0] - x_min) // leaf_size
        hy = (point_cloud[i][1] - y_min) // leaf_size
        hz = (point_cloud[i][2] - z_min) // leaf_size
        h.append(hx + hy * Dx + hz * Dx * Dy)
    h = np.array(h)

    # 筛选点

    min_vec = np.array([x_min, y_min, z_min])
    index = np.floor((point_cloud.copy() - min_vec) / leaf_size)
    h_index = index[:, 0] + index[:, 1] * Dx + index[:, 2] * Dx * Dy

    for index in np.unique(h_index):
        point_choosed = point_cloud[h_index == index]
        if if_mean:
            filtered_points.append(np.mean(point_choosed, axis=0))
        else:
            filtered_points.append(point_choosed[np.random.choice(a=point_choosed.shape[0])])
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    pc_txt_path = "/Users/hangz/PycharmProjects/pc/guitar_0060.txt"
    raw_data = np.genfromtxt(pc_txt_path, delimiter=",")
    point_cloud_data = raw_data[:, 0:3]
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_data)

    # 转成open3d能识别的格式
    #point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    points = np.asarray(point_cloud_o3d.points)
    filtered_cloud = voxel_filter(points, 0.05, if_mean=False)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
