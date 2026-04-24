import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    """将Euler角转换为旋转矩阵"""
    batch_size = euler_angles.shape[0]
    theta = euler_angles[:, 0]
    phi = euler_angles[:, 1]
    psi = euler_angles[:, 2]
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_psi = torch.cos(psi)
    sin_psi = torch.sin(psi)
    
    if convention == "XYZ":
        R = torch.zeros((batch_size, 3, 3), device=euler_angles.device)
        R[:, 0, 0] = cos_phi * cos_psi
        R[:, 0, 1] = -cos_phi * sin_psi
        R[:, 0, 2] = sin_phi
        R[:, 1, 0] = cos_theta * sin_psi + sin_theta * sin_phi * cos_psi
        R[:, 1, 1] = cos_theta * cos_psi - sin_theta * sin_phi * sin_psi
        R[:, 1, 2] = -sin_theta * cos_phi
        R[:, 2, 0] = sin_theta * sin_psi - cos_theta * sin_phi * cos_psi
        R[:, 2, 1] = sin_theta * cos_psi + cos_theta * sin_phi * sin_psi
        R[:, 2, 2] = cos_theta * cos_phi
    else:
        raise NotImplementedError("Only XYZ convention is supported")
    
    return R

class BundleAdjustment:
    def __init__(self, points2d_path, points3d_colors_path, image_width=1024, image_height=1024):
        # 加载数据
        self.points2d = np.load(points2d_path)
        self.points3d_colors = np.load(points3d_colors_path)
        self.image_width = image_width
        self.image_height = image_height
        
        # 相机中心点
        self.cx = image_width / 2
        self.cy = image_height / 2
        
        # 视角数量和点数量
        self.num_views = len(self.points2d.keys())
        self.num_points = len(self.points3d_colors)
        
        # 初始化参数
        self.initialize_parameters()
        
        # 准备数据
        self.prepare_data()
    
    def initialize_parameters(self):
        """初始化参数：焦距f，相机外参（旋转和平移），3D点坐标"""
        # 焦距初始化（假设FoV为60度）
        fov = 60 * np.pi / 180  # 转换为弧度
        self.f = torch.tensor([self.image_height / (2 * np.tan(fov / 2))], requires_grad=True)
        
        # 相机外参初始化：使用Euler角参数化旋转
        # 旋转初始化为单位矩阵（Euler角为0）
        self.euler_angles = torch.zeros((self.num_views, 3), requires_grad=True)
        # 平移初始化为[0, 0, -2.5]（相机在物体前方2.5单位）
        self.translations = torch.zeros((self.num_views, 3), requires_grad=True)
        with torch.no_grad():
            self.translations[:, 2] = -2.5
        
        # 3D点坐标初始化：在原点附近的随机位置
        self.points3d = torch.randn((self.num_points, 3), requires_grad=True) * 0.1
    
    def prepare_data(self):
        """准备训练数据"""
        self.views = []
        self.observed_points = []
        self.visibility = []
        
        for i, key in enumerate(self.points2d.keys()):
            data = self.points2d[key]
            self.views.append(i)
            self.observed_points.append(torch.tensor(data[:, :2], dtype=torch.float32))
            self.visibility.append(torch.tensor(data[:, 2], dtype=torch.bool))
    
    def project_points(self, points3d, euler_angles, translations, f):
        """将3D点投影到2D像素坐标"""
        # 计算旋转矩阵
        R = euler_angles_to_matrix(euler_angles, convention="XYZ")  # (num_views, 3, 3)
        
        # 相机坐标系中的点
        # points3d: (num_views, num_points, 3)
        # R: (num_views, 3, 3)
        # 矩阵乘法：(num_views, 3, 3) @ (num_views, 3, num_points) -> (num_views, 3, num_points)
        Xc = torch.matmul(R, points3d.permute(0, 2, 1)).permute(0, 2, 1) + translations  # (num_views, num_points, 3)
        
        # 投影到像素坐标
        u = -f * Xc[..., 0] / Xc[..., 2] + self.cx
        v = f * Xc[..., 1] / Xc[..., 2] + self.cy
        
        return torch.stack([u, v], dim=-1)  # (num_views, num_points, 2)
    
    def compute_loss(self):
        """计算重投影误差"""
        loss = 0.0
        
        # 对每个视角计算重投影误差
        for i in range(self.num_views):
            # 投影3D点
            projected = self.project_points(
                self.points3d.unsqueeze(0),  # 添加视角维度
                self.euler_angles[i].unsqueeze(0),
                self.translations[i].unsqueeze(0),
                self.f
            )[0]  # 移除视角维度
            
            # 只考虑可见点
            visible = self.visibility[i]
            if visible.sum() > 0:
                observed = self.observed_points[i][visible]
                predicted = projected[visible]
                
                # 计算均方误差
                loss += torch.mean(torch.sum((predicted - observed) ** 2, dim=1))
        
        return loss
    
    def optimize(self, num_epochs=1000, lr=0.01):
        """使用Adam优化器进行梯度下降"""
        # 确保所有参数都是叶子节点张量
        params = []
        for param_name, param in [
            ('f', self.f),
            ('euler_angles', self.euler_angles),
            ('translations', self.translations),
            ('points3d', self.points3d)
        ]:
            if not param.is_leaf:
                print(f"Warning: {param_name} is not a leaf tensor, making a copy")
                param = param.clone().detach().requires_grad_(True)
                setattr(self, param_name, param)
            params.append(param)
        
        # 优化器
        optimizer = optim.Adam(params, lr=lr)
        
        # 记录loss
        losses = []
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            optimizer.step()
            
            # 记录loss
            losses.append(loss.item())
            
            # 每100个epoch打印一次
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        return losses
    
    def save_point_cloud(self, output_path):
        """保存带颜色的3D点云为OBJ文件"""
        with open(output_path, 'w') as f:
            for i in range(self.num_points):
                x, y, z = self.points3d[i].detach().numpy()
                r, g, b = self.points3d_colors[i]
                f.write(f"v {x} {y} {z} {r} {g} {b}\n")
        print(f"Point cloud saved to {output_path}")
    
    def visualize_loss(self, losses, output_path):
        """可视化loss变化曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)), losses)
        plt.title('Bundle Adjustment Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(output_path)
        print(f"Loss curve saved to {output_path}")

if __name__ == "__main__":
    # 初始化Bundle Adjustment
    ba = BundleAdjustment(
        points2d_path="data/points2d.npz",
        points3d_colors_path="data/points3d_colors.npy"
    )
    
    # 优化
    print("Starting optimization...")
    losses = ba.optimize(num_epochs=1000, lr=0.01)
    
    # 可视化loss
    ba.visualize_loss(losses, "loss_curve.png")
    
    # 保存点云
    ba.save_point_cloud("reconstructed_point_cloud.obj")
    
    print("Bundle Adjustment completed!")
