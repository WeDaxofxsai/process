import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

"""
关节数据转xyz+四元数
"""



class DHParameterRobotOpen3D:
    def __init__(self, dh_params):
        """
        初始化DH参数机器人
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)
        
        # 坐标系6到坐标系8的变换参数
        self.frame6_to_frame8_transform = {
            'translation': [2.1606/1000, 2.1134/1000, 468.3159/1000],  # 转换为米
            'rotation': [-0.0075, -0.537, 3.1246]  # roll, pitch, yaw (弧度)
        }
        
    def dh_transform(self, a, alpha, d, theta):
        """
        计算单个DH参数的变换矩阵
        """
        T = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        return T
    
    def get_frame6_to_frame8_transform_matrix(self):
        """
        计算坐标系6到坐标系8的变换矩阵
        """
        # 提取变换参数
        tx, ty, tz = self.frame6_to_frame8_transform['translation']
        roll, pitch, yaw = self.frame6_to_frame8_transform['rotation']
        
        # RPY旋转矩阵
        rotation = R.from_euler('XYZ', [roll, pitch, yaw])
        R_matrix = rotation.as_matrix()
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = R_matrix
        T[:3, 3] = [tx, ty, tz]
        # # 创建绕Z轴旋转90度的旋转矩阵
        # angle_degrees = 90
        # theta = np.radians(angle_degrees)  # 将角度转换为弧度[2](@ref)

        # # 绕Z轴的基本旋转矩阵[1,6](@ref)
        # R_z = np.array([
        #     [np.cos(theta), -np.sin(theta), 0, 0],
        #     [np.sin(theta), np.cos(theta), 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])

        # # 将T绕自身Z轴旋转90度：通过矩阵乘法实现[7](@ref)
        # T = T @ R_z

        return T
    
    def forward_kinematics(self, joint_angles=None):
        """
        计算正向运动学
        """
        if joint_angles is None:
            joint_angles = [0.0] * self.num_joints
        
        transforms = [np.eye(4)]  # 基座变换矩阵
        positions = [np.array([0.0, 0.0, 0.0])]  # 基座位置
        
        # 计算每个关节的变换
        for i in range(self.num_joints):
            param = self.dh_params[i]
            T = self.dh_transform(
                param['a'], 
                param['alpha'], 
                param['d'], 
                joint_angles[i] + param['theta']
            )
            
            # 累积变换
            T_total = transforms[-1] @ T
            transforms.append(T_total)
            
            # 提取位置
            position = T_total[:3, 3]
            positions.append(position)
        
        # 计算坐标系8的变换矩阵
        T_6_8 = self.get_frame6_to_frame8_transform_matrix()
        T_0_8 = transforms[-1] @ T_6_8  # 从基座到坐标系8的变换
        frame8_transform = T_0_8
        
        return transforms, positions, frame8_transform


    # 下面都是可视化函数。
    def create_robot_geometry(self, joint_angles=None, show_frames=True, show_frame8=True):
        """
        创建机器人几何体
        """
        # 计算正向运动学
        transforms, positions, frame8_transform = self.forward_kinematics(joint_angles)
        frame8_position = frame8_transform[:3, 3]  # 提取位置向量
        print(frame8_transform)
        geometries = []
        # 创建连杆
        for i in range(len(positions) - 1):
            start = positions[i]
            end = positions[i+1]
            vec = end - start
            length = np.linalg.norm(vec)
            
            if length > 1e-6:
                # 创建圆柱体表示连杆
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.01, 
                    height=length
                )
                # 计算旋转使圆柱体朝向正确方向
                z_axis = np.array([0, 0, 1])
                rot_axis = np.cross(z_axis, vec / length)
                if np.linalg.norm(rot_axis) > 1e-6:
                    rot_angle = np.arccos(np.dot(z_axis, vec / length))
                    rot_matrix = R.from_rotvec(rot_axis * rot_angle).as_matrix()
                    cylinder.rotate(rot_matrix, center=[0, 0, 0])
                # 平移圆柱体
                cylinder.translate(start + vec / 2)
                cylinder.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
                geometries.append(cylinder)
        # 创建坐标系
        if show_frames:
            # 显示所有关节坐标系（0-6，共7个）
            for i in range(len(transforms)):
                transform = transforms[i]
                # 根据坐标系类型设置不同的轴长
                if i == 0:  # 基座坐标系
                    axis_length = 0.15
                elif i == len(transforms)-1:  # 末端坐标系（坐标系6）
                    axis_length = 0.12
                else:  # 中间关节坐标系
                    axis_length = 0.1
                # 创建坐标轴（红绿蓝三色）
                frame = self.create_coordinate_frame(transform, axis_length)
                geometries.append(frame)
            # 显示坐标系8
            if show_frame8:
                axis_length = 0.1
                frame8_frame = self.create_coordinate_frame(frame8_transform, axis_length)
                geometries.append(frame8_frame)

        return geometries
    
    def create_coordinate_frame(self, transform, axis_length=0.1):
        """
        创建坐标系（红绿蓝三色）
        """
        # 创建坐标轴
        points = [transform[:3, 3]]  # 原点
        lines = []
        colors = []
        
        # 标准红绿蓝三色
        axis_colors = [
            [1.0, 0.0, 0.0],  # 红色 - X轴
            [0.0, 1.0, 0.0],  # 绿色 - Y轴
            [0.0, 0.0, 1.0]   # 蓝色 - Z轴
        ]
        
        # X轴（红色）
        x_end = transform[:3, 3] + transform[:3, 0] * axis_length
        points.append(x_end)
        lines.append([0, 1])
        colors.append(axis_colors[0])
        
        # Y轴（绿色）
        y_end = transform[:3, 3] + transform[:3, 1] * axis_length
        points.append(y_end)
        lines.append([0, 2])
        colors.append(axis_colors[1])
        
        # Z轴（蓝色）
        z_end = transform[:3, 3] + transform[:3, 2] * axis_length
        points.append(z_end)
        lines.append([0, 3])
        colors.append(axis_colors[2])
        
        # 创建LineSet
        frame = o3d.geometry.LineSet()
        frame.points = o3d.utility.Vector3dVector(points)
        frame.lines = o3d.utility.Vector2iVector(lines)
        frame.colors = o3d.utility.Vector3dVector(colors)
        
        return frame


    def joints_to_cartesian_7d(self, joint_angles):
        """
        将关节角度转换为7维笛卡尔空间坐标（位置+四元数）
        
        参数:
            joint_angles: 关节角度列表（弧度），长度应为6
            
        返回:
            7维向量 [x, y, z, qx, qy, qz, qw]
        """
        # 计算正向运动学
        _, _, frame8_transform = self.forward_kinematics(joint_angles)
        
        # 提取位置 (x, y, z)
        position = frame8_transform[:3, 3]*1000
        
        # 提取旋转矩阵并转换为四元数
        rotation_matrix = frame8_transform[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # 返回 [qx, qy, qz, qw]
        
        # 组合成7维向量: [x, y, z, qx, qy, qz, qw]
        cartesian_7d = np.concatenate([position, quaternion])
        
        return cartesian_7d

    def convert_joint_file_to_cartesian(self, input_file, output_file, precision=6):
        """
        读取关节空间文件，转换为笛卡尔空间文件
        
        参数:
            input_file: 关节角度文件路径，每行6个关节角度
            output_file: 输出文件路径
            precision: 小数点后位数
        """
        joint_trajectory = []
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                values = line.split(' ')
                try:
                    # 转换为浮点数，取前6个值
                    joints = [float(val) for val in values[:6]]
                    if len(joints) == 6:
                        joint_trajectory.append(joints)
                    else:
                        print(f"警告: 第{line_num}行只有{len(joints)}个值，跳过")
                except ValueError as e:
                    print(f"警告: 第{line_num}行包含无效数据，跳过: {line}")
                    continue
        
        
        # 转换为笛卡尔坐标
        cartesian_trajectory = []
        for i, joints in enumerate(joint_trajectory, 1):
            cartesian_7d = self.joints_to_cartesian_7d(joints)
            cartesian_trajectory.append(cartesian_7d)
            
        res = np.array(cartesian_trajectory)

        res = np.hstack((res, np.ones((len(res), 1))))
        time_col = np.arange(1, len(res)+1).reshape(-1, 1)
        res = np.hstack((res, time_col))

        np.savetxt('util.txt', res, delimiter=',', fmt='%.6f')

        return True

def main():
    # 定义DH参数
    dh_params = [
        {'joint': 1, 'a': 0.110873, 'alpha': 1.5707963267949, 'd': 0.4111, 'theta': 0.0},
        {'joint': 2, 'a': 0.830254, 'alpha': 0.0, 'd': 0.0061234, 'theta': 1.5707963267949},
        {'joint': 3, 'a': 0.207239, 'alpha': 1.5707963267949, 'd': 0.0, 'theta': 0.0},
        {'joint': 4, 'a': 0.0, 'alpha': 1.5707963267949, 'd': 1.01866, 'theta': 0.0},
        {'joint': 5, 'a': 0.0, 'alpha': -1.5707963267949, 'd': 0.0, 'theta': 1.5707963267949},
        {'joint': 6, 'a': 0.0, 'alpha': 0.0, 'd': 0.1015, 'theta': 0.0}
    ]
    
    # 创建机器人实例
    robot = DHParameterRobotOpen3D(dh_params)
    robot.convert_joint_file_to_cartesian("/home/zq/Downloads/data/processed_joints.txt", "util.txt")



if __name__ == "__main__":
    main()