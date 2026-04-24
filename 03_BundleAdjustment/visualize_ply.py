import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct

# 读取PLY文件（支持二进制格式）
def read_ply_file(file_path):
    points = []
    colors = []
    
    with open(file_path, 'rb') as f:
        # 读取头部信息
        header = b''
        while b'end_header' not in header:
            line = f.readline()
            if not line:
                break
            header += line
        
        # 解析头部信息
        header_str = header.decode('ascii')
        lines = header_str.split('\n')
        
        # 查找顶点数量
        vertex_count = 0
        for line in lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
                break
        
        # 读取点数据
        for _ in range(vertex_count):
            # 读取X, Y, Z坐标（float32）
            x = struct.unpack('f', f.read(4))[0]
            y = struct.unpack('f', f.read(4))[0]
            z = struct.unpack('f', f.read(4))[0]
            
            # 读取R, G, B颜色（unsigned char）
            r = struct.unpack('B', f.read(1))[0] / 255.0
            g = struct.unpack('B', f.read(1))[0] / 255.0
            b = struct.unpack('B', f.read(1))[0] / 255.0
            
            points.append([x, y, z])
            colors.append([r, g, b])
    
    return np.array(points), np.array(colors)

# 可视化点云
def visualize_point_cloud(points, colors, point_size=1):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置标题
    ax.set_title('COLMAP Dense Reconstruction')
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    plt.show()

if __name__ == "__main__":
    # 读取PLY文件
    ply_file = 'data/colmap/dense/fused.ply'
    print(f"Reading {ply_file}...")
    points, colors = read_ply_file(ply_file)
    
    print(f"Loaded {len(points)} points")
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")
    
    # 可视化点云
    print("Visualizing point cloud...")
    visualize_point_cloud(points, colors, point_size=1)
