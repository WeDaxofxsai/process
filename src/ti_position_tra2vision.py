import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import math
import copy

"""
笛卡尔空间运动轨迹可视化

"""



class DHParameterRobotOpen3D:
    def __init__(self, dh_params):
        """
        初始化DH参数机器人
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)
        
        # 关节限位
        self.joint_limits = [
            {'min': -np.pi, 'max': np.pi},  # 关节1
            {'min': -2.356, 'max': 1.571},  # 关节2
            {'min': -1.343, 'max': 3.543},  # 关节3
            {'min': -2.443, 'max': 2.443},  # 关节4
            {'min': -3.665, 'max': 0.279},  # 关节5
            {'min': -2*np.pi, 'max': 2*np.pi}   # 关节6
        ]

        # 坐标系6到坐标系8的变换参数
        self.frame6_to_frame8_transform = {
            'translation': [2.1606/1000, 2.1134/1000, 468.3159/1000],  # 转换为米
            'rotation': [-0.0075, -0.537, 3.1246]  # roll, pitch, yaw (弧度)
        }
        
        # 逆解参数
        self.inverse_params = {
            'max_iterations': 100,
            'position_tolerance': 0.001,  # 1mm，kun_changed = 0.001
            'orientation_tolerance': 0.01,  # 约0.57度
            'lambda_val': 0.1,  # 阻尼因子
            'joint_weight_A': 1,  # A部分权重倍率
            'joint_weight_B': 5.0,   # B部分权重倍率
            'distance_threshold_large': 0.1,  # 大距离阈值   kun_changed = 0.1
            'distance_threshold_small': 0.005,  # 小距离阈值  kun_changed = 0.01
            'safe_angles': [(self.joint_limits[0]['min']+self.joint_limits[0]['max'])/2 ,  (self.joint_limits[1]['min']+self.joint_limits[1]['max'])/2, 
                            (self.joint_limits[2]['min']+self.joint_limits[2]['max'])/2 , (self.joint_limits[3]['min']+self.joint_limits[3]['max'])/2, 
                            (self.joint_limits[4]['min']+self.joint_limits[4]['max'])/2, (self.joint_limits[5]['min']+self.joint_limits[5]['max'])/2]  # 理想关节角（中间位置）
        }
        
        # 当前关节状态
        self.current_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
          # 添加这一行：存储目标点位姿的列表
        self.target_poses = []
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
        # 创建绕Z轴旋转90度的旋转矩阵
        angle_degrees = 90
        theta = np.radians(angle_degrees)  # 将角度转换为弧度[2](@ref)

        # 绕Z轴的基本旋转矩阵[1,6](@ref)
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 将T绕自身Z轴旋转90度：通过矩阵乘法实现[7](@ref)
        T = T @ R_z
        return T
    
    def get_frame8_to_frame6_transform_matrix(self):
        """
        计算坐标系8到坐标系6的变换矩阵（逆变换）
        """
        T_6_8 = self.get_frame6_to_frame8_transform_matrix()
        # print("T_6_8",T_6_8)
        return np.linalg.inv(T_6_8)
    
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

    def jacobian(self, joint_angles):
        """
        计算几何雅可比矩阵
        """
        transforms, positions, _ = self.forward_kinematics(joint_angles)
        jac = np.zeros((6, self.num_joints))
        
        # 末端位置
        p_end = positions[-1]
        
        for i in range(self.num_joints):
            # 关节i的z轴方向
            z_axis = transforms[i][:3, 2]
            
            # 位置部分：z_i × (p_end - p_i)
            p_i = positions[i]
            jac[:3, i] = np.cross(z_axis, p_end - p_i)
            
            # 姿态部分：z_i
            jac[3:, i] = z_axis
            
        return jac
    
    def calculate_error(self, T_current, T_target):
        """
        计算当前位姿与目标位姿之间的误差
        """
        # 位置误差
        pos_error = T_target[:3, 3] - T_current[:3, 3]
        
        # 姿态误差（轴角表示）
        R_current = T_current[:3, :3]
        R_target = T_target[:3, :3]
        R_error = R_target @ R_current.T
        rotvec = R.from_matrix(R_error).as_rotvec()
        
        # 组合误差向量
        error = np.concatenate([pos_error, rotvec])
        
        return error
    
    def adjust_weights(self, joint_errors, position_distance):
        """
        调整关节权重（核心优化部分）
        """
        weights = np.ones(self.num_joints)
        
        # A部分：当前关节状态与理想关节的差异
        safe_angles = self.inverse_params['safe_angles']
        joint_weight_A = self.inverse_params['joint_weight_A']
        
        for i in range(self.num_joints):
            # 计算关节i的危险系数
            angle_diff = abs(self.current_joints[i] - safe_angles[i])
            joint_range = self.joint_limits[i]['max'] - self.joint_limits[i]['min']
            
            # 归一化危险系数（0-1）
            danger_coefficient = angle_diff / (joint_range / 2) if joint_range > 0 else 0
            
            # 限制在0-1之间
            danger_coefficient = min(max(danger_coefficient, 0), 1)
            
            # A部分权重（危险系数高 → 权重高 → 该关节移动量小）
            weight_A = danger_coefficient * joint_weight_A
            weights[i] += weight_A
        
        # B部分：根据距离调整关节1-3和4-6的权重
        joint_weight_B = self.inverse_params['joint_weight_B']
        distance_threshold_large = self.inverse_params['distance_threshold_large']
        distance_threshold_small = self.inverse_params['distance_threshold_small']
        
        if position_distance > distance_threshold_large:
            # 大距离：关节1-3权重低（移动量大），关节4-6权重高（移动量小）
            weights[0:2] *= 1.0  # 关节1-3保持较低权重
            weights[4:6] *= 2.0  # 关节4-6增加权重
        elif position_distance < distance_threshold_small:
            # 小距离：关节4-6权重低（移动量大），关节1-3权重高（移动量小）
            weights[0:2] *= 2.0  # 关节1-3增加权重
            weights[4:6] *= 1.0  # 关节4-6保持较低权重
        
        return weights
    
    def inverse_kinematics(self, target_T_6, initial_joints=None, max_iterations=None):
        """
        改进的逆运动学算法（带权重调整）
        """
        if initial_joints is None:
            initial_joints = self.current_joints.copy()
        
        if max_iterations is None:
            max_iterations = self.inverse_params['max_iterations']
            #         'max_iterations': 100,
            # 'position_tolerance': 0.001,  # 1mm
            # 'orientation_tolerance': 0.01,  # 约0.57度
            # 'lambda_val': 0.1,  # 阻尼因子
            # 'joint_weight_A': 5,  # A部分权重倍率
            # 'joint_weight_B': 5.0,   # B部分权重倍率
            # 'distance_threshold_large': 0.1,  # 大距离阈值
            # 'distance_threshold_small': 0.01,  # 小距离阈值
        joints = initial_joints.copy()
        lambda_val = self.inverse_params['lambda_val']
        pos_tol = self.inverse_params['position_tolerance']
        rot_tol = self.inverse_params['orientation_tolerance']
        
        # 记录最佳解
        best_joints = joints.copy()

        best_error = float('inf')
        
        for iteration in range(max_iterations):
            # 计算当前正运动学
            transforms, _, _ = self.forward_kinematics(joints)
            T_current = transforms[-1]
            
            # 计算误差
            error = self.calculate_error(T_current, target_T_6)
            pos_error_norm = np.linalg.norm(error[:3])
            rot_error_norm = np.linalg.norm(error[3:])
            
            # 更新最佳解
            total_error = pos_error_norm + rot_error_norm * 0.1
            # print(iteration,total_error)
            if total_error < best_error:
                best_joints = joints.copy()
                best_error = total_error
            
            # 检查是否收敛
            if pos_error_norm < pos_tol and rot_error_norm < rot_tol:
                # print(f"逆解收敛于迭代 {iteration}")
                # 更新当前关节状态
                self.current_joints = joints.copy()
                return joints
            
            # 计算雅可比矩阵
            J = self.jacobian(joints)
            
            # 计算权重
            # print("error",error)
            weights = self.adjust_weights(error, pos_error_norm)
            # print("weights",weights)
            W = np.diag(weights)
            
            # 计算加权阻尼最小二乘解
            J_weighted = J @ np.linalg.inv(W)
            delta_theta = J_weighted.T @ np.linalg.inv(
                J_weighted @ J_weighted.T + lambda_val * np.eye(6)
            ) @ error
            # 更新关节角度
            joints += delta_theta
            # 关节限位处理
            for i in range(self.num_joints):
                joint_min = self.joint_limits[i]['min']
                joint_max = self.joint_limits[i]['max']
                
                # 角度归一化到[-pi, pi]
                while joints[i] > np.pi:
                    joints[i] -= 2 * np.pi
                while joints[i] < -np.pi:
                    joints[i] += 2 * np.pi
                
                # 检查限位 
                if joints[i] < joint_min:
                    joints[i] = joint_min
                elif joints[i] > joint_max:
                    joints[i] = joint_max

        print(f"逆解未完全收敛，使用最佳解，位置误差: {best_error:.6f}")
        # 更新当前关节状态
        self.current_joints = best_joints.copy()
        return best_joints
    
    def trajectory_point_to_frame6(self, point_xyz_xyzw):
        """
        将坐标系8的轨迹点转换为坐标系6的位姿
        """
        # 解析输入：xyz + xyzw + flag
        x, y, z, qx, qy, qz, qw, flag = point_xyz_xyzw
        
        if flag == 0:
            return None  # 跳过此点
        
        # 构建坐标系8的变换矩阵
        T_8 = np.eye(4)
        T_8[:3, 3] = [x, y, z]
        
        # 四元数转旋转矩阵
        rotation = R.from_quat([qx, qy, qz, qw])
        T_8[:3, :3] = rotation.as_matrix()
        
        # 获取坐标系8到6的变换矩阵
        T_8_6 = self.get_frame8_to_frame6_transform_matrix()
        self.target_poses.append(T_8.copy())
        # print("T_8_6",T_8_6)
        # print("T_8",T_8)
        # 计算坐标系6的变换矩阵
        T_6 = T_8 @ T_8_6
        # 添加这一行：将目标位姿保存到列表
        self.target_poses.append(T_6.copy())
        return T_6
    
    def linear_interpolation(self, start_joints, target_joints, num_steps=10):
        """
        在关节空间进行线性插值
        """
        if num_steps <= 1:
            return [target_joints]
        
        interpolated = []
        for i in range(1, num_steps + 1):
            t = i / num_steps
            interp_joints = start_joints * (1 - t) + target_joints * t
            interpolated.append(interp_joints)
        
        return interpolated
    
    def cartesian_interpolation(self, start_T, target_T, num_steps=10):
        """
        在笛卡尔空间进行插值
        """
        if num_steps <= 1:
            return [target_T]
        
        interpolated = []
        # 位置插值
        start_pos = start_T[:3, 3]
        target_pos = target_T[:3, 3]
        
        # 姿态插值（四元数）
        start_rot = R.from_matrix(start_T[:3, :3])
        target_rot = R.from_matrix(target_T[:3, :3])
        
        for i in range(1, num_steps + 1):
            t = i / num_steps
            # 位置插值
            interp_pos = start_pos * (1 - t) + target_pos * t
            
            # 姿态插值（球面线性插值）
            interp_rot = R.slerp(start_rot, target_rot, t)
            
            # 构建变换矩阵
            T = np.eye(4)
            T[:3, 3] = interp_pos
            T[:3, :3] = interp_rot.as_matrix()
            
            interpolated.append(T)
        
        return interpolated
    
    def execute_trajectory(self, trajectory_file, enable_interpolation=True, interpolation_threshold=0.1):
        """
        执行轨迹文件
        """
        self.target_poses = []
        # 读取轨迹文件
        with open(trajectory_file, 'r') as f:
            lines = f.readlines()
        
        # 解析轨迹点
        trajectory_points = []
        for line in lines:
            if line.strip():
                values = list(map(float, line.strip().split(',')))
                if len(values)  == 9:  # xyz + xyzw + flag + v
                    trajectory_points.append([values[0]/1000,values[1]/1000,values[2]/1000,values[3],values[4],values[5],values[6],values[7]])
        
        # 执行轨迹
        all_joint_sequences = []
        
        for i, point in enumerate(trajectory_points):
            # print(f"\n处理第 {i+1}/{len(trajectory_points)} 个轨迹点...")
            # print("变换前",point)
            # 转换到坐标系6
            T_6_target = self.trajectory_point_to_frame6(point)

            # print("变换后",T_6_target)
            
            # 计算逆解
            if i == 0:
                # 第一个点：从当前位置开始
                start_joints = self.current_joints.copy()
            else:
                start_joints = all_joint_sequences[-1][-1].copy()
            
            target_joints = self.inverse_kinematics(T_6_target, start_joints)
            # print("target_joints",target_joints)
            
            # 检查是否需要插值
            if enable_interpolation:
                # 计算当前位置与目标位置的距离
                transforms, _, _ = self.forward_kinematics(start_joints)
                T_current = transforms[-1]
                current_pos = T_current[:3, 3]
                target_pos = T_6_target[:3, 3]

                position_distance = np.linalg.norm(target_pos - current_pos)
                
                # print(f"位置距离: {position_distance:.4f}m, 阈值: {interpolation_threshold}m")
                
                if position_distance > interpolation_threshold:
                    # 需要插值
                    # print(f"进行插值...")
                    num_steps = int(position_distance / 0.02)  # 每2cm一个插值点
                    
                    # 关节空间插值
                    interpolated_joints = self.linear_interpolation(start_joints, target_joints, num_steps)
                    
                    # 添加插值后的关节序列
                    for interp_joint in interpolated_joints:
                        all_joint_sequences.append([interp_joint])
                else:
                    # 不需要插值
                    all_joint_sequences.append([target_joints])
            else:
                # 不需要插值
                all_joint_sequences.append([target_joints])
            
            # 更新当前关节状态
            self.current_joints = target_joints.copy()
        
        # 展平关节序列
        flattened_sequence = []
        for sequence in all_joint_sequences:
            flattened_sequence.extend(sequence)
        
        return flattened_sequence
    
    def visualize_trajectory(self, joint_sequence, frame_delay=0.1, show_progress=True):
        """
        可视化轨迹执行过程（改进版，带固定视角）
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="机器人轨迹执行",
            width=1200,
            height=800,
            left=50,
            top=50,
            visible=True
        )
        
        # 创建初始几何体并保存引用
        robot_geometries = self.create_robot_geometry(
            joint_angles=joint_sequence[0],
            show_frames=True,
            show_frame8=True
        )
        
        # 添加这里：创建目标点位姿坐标系（白色）
        target_frames = []
        for i, target_pose in enumerate(self.target_poses):
            # 白色坐标系
            target_frame = self.create_target_coordinate_frame(target_pose, axis_length=0.15, color=[1.0, 1.0, 1.0])
            
            target_frames.append(target_frame)


        # 保存所有几何体的引用
        geometry_refs = {}
        for idx, geom in enumerate(robot_geometries):
            vis.add_geometry(geom)
            geometry_refs[idx] = geom
        # 添加这里：添加目标坐标系到可视化器
        for frame in target_frames:
            vis.add_geometry(frame)
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 3.0
        opt.line_width = 2.0
        opt.light_on = True
        opt.show_coordinate_frame = False
        
        # 设置初始视角
        ctr = vis.get_view_control()
        ctr.set_front([0.5, -0.5, 0.5])
        ctr.set_lookat([0.5, 0, 0.5])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.8)
        
        # 保存相机参数
        initial_camera_params = ctr.convert_to_pinhole_camera_parameters()
        
        # 轨迹可视化
        print("\n开始轨迹可视化...")
        print(f"总步数: {len(joint_sequence)}")
        print("按'Q'或关闭窗口结束")
        print("按'S'减速，按'F'加速")
        
        # 初始化
        vis.poll_events()
        vis.update_renderer()
        
        current_delay = frame_delay
        pause = False
        
        for i, joints in enumerate(joint_sequence):
            if not vis.poll_events():
                break
                
            # 检查按键
            if hasattr(vis, 'get_key_pressed'):
                key = vis.get_key_pressed()
                if key == ord('S'):  # 减速
                    current_delay = min(1.0, current_delay + 0.05)
                    print(f"减速: 延迟 {current_delay:.2f}秒")
                elif key == ord('F'):  # 加速
                    current_delay = max(0.01, current_delay - 0.05)
                    print(f"加速: 延迟 {current_delay:.2f}秒")
                elif key == ord('P'):  # 暂停/继续
                    pause = not pause
                    print(f"{'暂停' if pause else '继续'}")
                elif key == ord('R'):  # 重置速度
                    current_delay = frame_delay
                    print(f"重置速度: 延迟 {current_delay:.2f}秒")
                elif key == ord('V'):  # 重置视角
                    ctr.convert_from_pinhole_camera_parameters(initial_camera_params)
                    print("视角已重置")
            
            if pause:
                # 等待继续
                time.sleep(0.1)
                continue
            
            # 显示进度
            if show_progress and i % max(1, len(joint_sequence)//20) == 0:
                progress = (i + 1) / len(joint_sequence) * 100
                print(f"进度: {progress:.1f}% ({i+1}/{len(joint_sequence)})")
            
            # 更新机器人姿态 - 重新创建几何体
            new_geometries = self.create_robot_geometry(
                joint_angles=joints,
                show_frames=True,
                show_frame8=True
            )
            
            # 清除旧几何体并添加新几何体
            vis.clear_geometries()
            
            for idx, geom in enumerate(new_geometries):
                vis.add_geometry(geom, reset_bounding_box=False)  # 不重置边界框
            
            # 添加这里：重新添加目标坐标系
            for frame in target_frames:
                vis.add_geometry(frame, reset_bounding_box=False)

            # 恢复相机参数以保持视角
            ctr.convert_from_pinhole_camera_parameters(initial_camera_params)
            
            # 更新视图
            vis.poll_events()
            vis.update_renderer()
            
            # 控制播放速度
            time.sleep(current_delay)
        
        # 保持窗口打开
        print("\n轨迹播放完成！")
        print("按任意键关闭窗口...")
        vis.run()
        vis.destroy_window()
    
    def smooth_trajectory(self, joint_sequence, window_size=5):
        """
        平滑轨迹（减少抖动）
        window_size: 平滑窗口大小
        """
        if len(joint_sequence) <= window_size:
            return joint_sequence
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(joint_sequence)):
            # 获取窗口内的关节角度
            start = max(0, i - half_window)
            end = min(len(joint_sequence), i + half_window + 1)
            
            # 计算移动平均
            window_joints = joint_sequence[start:end]
            avg_joints = np.mean(window_joints, axis=0)
            
            smoothed.append(avg_joints)
        
        return smoothed

    # 下面都是可视化函数（保持不变）
    def create_robot_geometry(self, joint_angles=None, show_frames=True, show_frame8=True):
        """
        创建机器人几何体
        """
        # 计算正向运动学
        transforms, positions, frame8_transform = self.forward_kinematics(joint_angles)
        frame8_position = frame8_transform[:3, 3]  # 提取位置向量
        
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
                if i == 0:
                    frame.colors = o3d.utility.Vector3dVector([
                                [0.0, 0.0, 1.0]  , # 蓝色 - Z轴
                                [0.0, 0.0, 1.0] ,  # 蓝色 - Z轴
                                [0.0, 0.0, 1.0]   # 蓝色 - Z轴
                            ])
                geometries.append(frame)

            # 显示坐标系8
            if show_frame8:
                axis_length = 0.1
                frame8_frame = self.create_coordinate_frame(frame8_transform, axis_length)
                frame8_frame.colors = o3d.utility.Vector3dVector([
                                [1.0, 0.0, 0.0]  , # 蓝色 - Z轴
                                [1.0, 0.0, 0.0] ,  # 蓝色 - Z轴
                                [1.0, 0.0, 0.0]   # 蓝色 - Z轴
                            ])
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
   
    def create_target_coordinate_frame(self, transform, axis_length=0.15, color=None):
        """
        创建目标点位姿坐标系（默认为白色）
        color: [R, G, B] 范围0-1
        """
        if color is None:
            color = [1.0, 1.0, 1.0]  # 白色
        
        # 创建坐标轴
        points = [transform[:3, 3]]  # 原点
        lines = []
        colors = []
        
        # 所有轴都使用相同的白色
        axis_colors = [
            color,  # X轴 - 白色
            color,  # Y轴 - 白色
            [1.0, 0.0, 1.0]   # Z轴 - 白色
        ]
        
        # X轴
        x_end = transform[:3, 3] + transform[:3, 0] * axis_length
        points.append(x_end)
        lines.append([0, 1])
        colors.append(axis_colors[0])
        
        # Y轴
        y_end = transform[:3, 3] + transform[:3, 1] * axis_length
        points.append(y_end)
        lines.append([0, 2])
        colors.append(axis_colors[1])
        
        # Z轴
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



def main():
    # 定义DH参数
    dh_params = [
        {'joint': 1, 'a': 0.110873, 'alpha': 1.5707963267949,       'd': 0.4111,        'theta': 0.0},
        {'joint': 2, 'a': 0.830254, 'alpha': 0.0,                   'd': 0.0061234,     'theta': 1.5707963267949},
        {'joint': 3, 'a': 0.207239, 'alpha': 1.5707963267949,       'd': 0.0,           'theta': 0.0},
        {'joint': 4, 'a': 0.0,      'alpha': 1.5707963267949,       'd': 1.01866,       'theta': 0.0},
        {'joint': 5, 'a': 0.0,      'alpha': -1.5707963267949,      'd': 0.0,           'theta': 1.5707963267949},
        {'joint': 6, 'a': 0.0,      'alpha': 0.0,                   'd': 0.1015,        'theta': 0.0}
    ]
    
    # 创建机器人实例
    robot = DHParameterRobotOpen3D(dh_params)
    
    # 执行轨迹
    # print("开始执行轨迹...")
    joint_sequence = robot.execute_trajectory(
        "./source/origin_position_tra.txt",
        enable_interpolation=False,
        interpolation_threshold=0.005
    )
    
    print(f"\n轨迹执行完成，共 {len(joint_sequence)} 个关节配置")
    
    # 平滑轨迹（可选）
    if len(joint_sequence) > 10:
        joint_sequence = robot.smooth_trajectory(joint_sequence, window_size=3)
        print("轨迹已平滑处理")
    
    # 可视化轨迹（带速度控制）
    # robot.visualize_trajectory(
    #     joint_sequence, 
    #     frame_delay=0.2,  # 初始延迟0.2秒
    #     show_progress=True
    # )
    # print(joint_sequence)
    np.savetxt("./source/test_joint.txt", joint_sequence ,delimiter= ',',fmt='%.6f')

if __name__ == "__main__":
    main()