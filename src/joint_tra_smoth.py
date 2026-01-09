import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class TeachPendantTrajectoryProcessor:
    def __init__(self, robot_model = None):
        self.robot_model = robot_model
        self.trajectory = None
        
    def process_pipeline(self, cartesian_points, downsample_rate=0.1):
        """
        完整处理流程
        """
        # 1. 下采样
        sampled_points = self.adaptive_downsample(
            cartesian_points, 
            min_distance=0.1,
            max_curvature_weight=0.5
        )
        
        # 2. 转换为关节空间
        joint_trajectory = self.to_joint_space(sampled_points)
        
        # 3. 关节空间拟合
        fitted_trajectory = self.fit_joint_trajectory(
            joint_trajectory,
            method='spline',
            smoothing=0.05
        )
        
        # 4. 生成平滑轨迹
        smooth_trajectory = self.generate_smooth_trajectory(
            fitted_trajectory,
            dt=0.01  # 10ms间隔
        )
        
        return smooth_trajectory
    
    def adaptive_downsample(self, points, min_distance, max_curvature_weight):
        """
        自适应下采样算法
        """
        if len(points) < 3:
            return points
            
        # 计算曲率
        curvatures = self.calculate_curvature(points)
        
        # 根据曲率加权采样
        sampled_indices = [0]
        for i in range(1, len(points)-1):
            distance = np.linalg.norm(
                points[i] - points[sampled_indices[-1]]
            )
            curvature_weight = curvatures[i] / np.max(curvatures)
            
            if (distance > min_distance or 
                curvature_weight > max_curvature_weight):
                sampled_indices.append(i)
                
        sampled_indices.append(len(points)-1)
        return points[sampled_indices]
    
    def fit_joint_trajectory(self, joint_trajectory, method='spline', **kwargs):
        """关节轨迹拟合"""
        time = np.linspace(0, 1, len(joint_trajectory))
        
        if method == 'spline':
            # 三次样条拟合
            splines = []
            for i in range(joint_trajectory.shape[1]):
                spline = CubicSpline(
                    time, 
                    joint_trajectory[:, i],
                    bc_type='natural'
                )
                splines.append(spline)
            return splines
            
        elif method == 'polynomial':
            # 多项式拟合
            polynomials = []
            degree = kwargs.get('degree', 5)
            for i in range(joint_trajectory.shape[1]):
                coeffs = np.polyfit(time, joint_trajectory[:, i], degree)
                polynomials.append(np.poly1d(coeffs))
            return polynomials
    
    def calculate_trajectory_quality(self, original, fitted):
        """计算轨迹拟合质量"""
        # 计算最大误差
        max_error = np.max(np.abs(original - fitted))
        
        # 计算平滑度（通过加速度变化）
        dt = 0.01
        original_acc = np.diff(original, n=2) / (dt**2)
        fitted_acc = np.diff(fitted, n=2) / (dt**2)
        smoothness = np.std(fitted_acc) / np.std(original_acc)
        
        return {
            'max_error': max_error,
            'smoothness_ratio': smoothness
        }
        
    def calculate_curvature(self, points):
        """离散点曲率计算方法（最常用）"""
        n = len(points)
        curvatures = np.zeros(n)
        
        # 对内部点计算曲率
        for i in range(1, n-1):
            p0 = points[i-1]
            p1 = points[i]
            p2 = points[i+1]
            
            # 计算向量
            v1 = p1 - p0
            v2 = p2 - p1
            
            # 向量长度
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-10 and v2_norm > 1e-10:
                # 单位向量
                u1 = v1 / v1_norm
                u2 = v2 / v2_norm
                
                # 计算夹角
                dot_product = np.clip(np.dot(u1, u2), -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # 曲率 = 夹角变化率
                curvatures[i] = angle / ((v1_norm + v2_norm) / 2)
        
        # 处理端点（使用相邻点的曲率）
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        
        return curvatures
    

if __name__ == "__main__":
    data = np.loadtxt('source/origin_position_tra.txt', delimiter=',')
    x = data[:, 0]  # 第一列: 923.310522, 923.624275, ...
    y = data[:, 1]  # 第二列: 233.607330, 230.104031, ...
    z = data[:, 2]  # 第三列: 1884.392972, 1882.412603, ...
    origin_points = np.array([x,y,z]).T
    # tools = TeachPendantTrajectoryProcessor()
    # point = tools.adaptive_downsample(
    #         origin_points, 
    #         min_distance=1,
    #         max_curvature_weight=0.5
    #     )
    # print(len(point))
