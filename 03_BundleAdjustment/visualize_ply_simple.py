import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct

# 读取PLY文件（支持二进制格式）
def read_ply_file(file_path, max_points=10000):
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
        
        # 计算采样步长
        step = max(1, vertex_count // max_points)
        
        # 读取点数据
        for i in range(vertex_count):
            # 读取X, Y, Z坐标（float32）
            x = struct.unpack('f', f.read(4))[0]
            y = struct.unpack('f', f.read(4))[0]
            z = struct.unpack('f', f.read(4))[0]
            
            # 读取R, G, B颜色（unsigned char）
            r = struct.unpack('B', f.read(1))[0] / 255.0
            g = struct.unpack('B', f.read(1))[0] / 255.0
            b = struct.unpack('B', f.read(1))[0] / 255.0
            
            # 每隔step个点采样一次
            if i % step == 0:
                points.append([x, y, z])
                colors.append([r, g, b])
    
    return np.array(points), np.array(colors)

# 保存点云为简化的OBJ文件
def save_simple_obj(points, colors, output_path):
    with open(output_path, 'w') as f:
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"v {x} {y} {z} {r} {g} {b}\n")
    print(f"Simple OBJ file saved to {output_path}")

if __name__ == "__main__":
    # 读取PLY文件
    ply_file = 'data/colmap/dense/fused.ply'
    print(f"Reading {ply_file}...")
    points, colors = read_ply_file(ply_file, max_points=5000)
    
    print(f"Loaded {len(points)} points (sampled)")
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")
    
    # 保存为简化的OBJ文件
    save_simple_obj(points, colors, 'colmap_dense_simple.obj')
    
    print("You can now open 'colmap_dense_simple.obj' in Blender to view the point cloud.")
