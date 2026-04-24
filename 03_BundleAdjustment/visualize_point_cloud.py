import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取OBJ文件
def read_obj_file(file_path):
    points = []
    colors = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 6:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    r = float(parts[4])
                    g = float(parts[5])
                    b = float(parts[6])
                    points.append([x, y, z])
                    colors.append([r, g, b])
    
    return np.array(points), np.array(colors)

# 可视化点云
def visualize_point_cloud(points, colors):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置标题
    ax.set_title('Reconstructed 3D Point Cloud')
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    plt.show()

if __name__ == "__main__":
    # 读取OBJ文件
    points, colors = read_obj_file('reconstructed_point_cloud.obj')
    
    print(f"Loaded {len(points)} points")
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")
    print(f"Color range: min={colors.min()}, max={colors.max()}")
    
    # 可视化点云
    visualize_point_cloud(points, colors)
