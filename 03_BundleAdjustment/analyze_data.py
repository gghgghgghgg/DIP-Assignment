import numpy as np

# 分析 points2d.npz
print("=== 分析 points2d.npz ===")
points2d = np.load("data/points2d.npz")
print(f"包含的键: {list(points2d.keys())}")
print(f"视角数量: {len(points2d.keys())}")

# 查看第一个视角的数据
sample_key = list(points2d.keys())[0]
sample_data = points2d[sample_key]
print(f"\n第一个视角 ({sample_key}) 的数据形状: {sample_data.shape}")
print(f"数据类型: {sample_data.dtype}")
print(f"前5个点的数据:\n{sample_data[:5]}")
print(f"最后5个点的数据:\n{sample_data[-5:]}")

# 统计可见点数量
visibility = sample_data[:, 2]
visible_count = np.sum(visibility == 1.0)
total_count = len(visibility)
print(f"\n可见点数量: {visible_count}/{total_count} ({visible_count/total_count*100:.2f}%)")

# 分析 points3d_colors.npy
print("\n=== 分析 points3d_colors.npy ===")
points3d_colors = np.load("data/points3d_colors.npy")
print(f"数据形状: {points3d_colors.shape}")
print(f"数据类型: {points3d_colors.dtype}")
print(f"前5个点的颜色:\n{points3d_colors[:5]}")
print(f"颜色值范围: min={points3d_colors.min()}, max={points3d_colors.max()}")
