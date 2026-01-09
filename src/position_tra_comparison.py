import copy
import math
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R

"""
对比两段轨迹

"""


# 可视化
def create_coordinate_frame(transform, size=200.0, label=""):
    """创建坐标系框架"""
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coord_frame.transform(transform)
    return coord_frame

def create_vector_arrow(origin, direction, color=(1, 0, 0), length=1.0, shaft_ratio=0.75, tip_ratio=0.25, radius=0.05):
    """
    创建一个完美对齐的3D箭头（圆柱 + 圆锥），永不分离！
    
    参数:
        origin: np.array([x,y,z]) 起点
        direction: np.array([x,y,z]) 方向向量（任意长度）
        color: RGB颜色 (1,0,0) 为红色
        length: 箭头总长度
        shaft_ratio: 箭身占比 (默认0.75)
        tip_ratio: 箭头占比 (默认0.25)
        radius: 箭身半径
    """
    origin = np.asarray(origin)
    direction = np.asarray(direction)
    
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([0, 0, 1])
    
    # 归一化方向并缩放到目标长度
    dir_normalized = direction / np.linalg.norm(direction)
    dir_scaled = dir_normalized * length
    
    # 箭身（圆柱）
    shaft_length = length * shaft_ratio
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=shaft_length, resolution=32, split=1
    )
    
    # 箭头（圆锥）
    cone = o3d.geometry.TriangleMesh.create_cone(
        radius=radius * 3, height=length * tip_ratio, resolution=32, split=1
    )
    
    # 关键：先把圆柱和圆锥都移动到原点，再旋转，最后整体平移到 origin
    # 1. 先把圆柱底部移到原点（create_cylinder 默认底部在原点）
    cylinder.translate(-cylinder.get_center() + np.array([0, 0, shaft_length / 2]))
    
    # 2. 圆锥底部移到原点（create_cone 默认底部在原点）
    cone.translate(-cone.get_center() + np.array([0, 0, length * tip_ratio / 2]))
    
    # 3. 把圆锥移动到圆柱顶部（无缝对接）
    cone.translate([0, 0, shaft_length])
    
    # 4. 对齐到目标方向（从 Z 轴 旋转到 direction）
    z_axis = np.array([0, 0, 1])
    if np.abs(np.dot(z_axis, dir_normalized)) < 0.999:
        axis = np.cross(z_axis, dir_normalized)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, dir_normalized))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(R, center=(0,0,0))
        cone.rotate(R, center=(0,0,0))
    
    # 5. 整体平移到起点
    cylinder.translate(origin)
    cone.translate(origin)
    
    # 上色
    cylinder.paint_uniform_color(color)
    cone.paint_uniform_color(color)
    
    return cylinder, cone

def visualize_with_open3d(world_coor_m_, second_box_m_, world_vector):
    """使用Open3D进行可视化（包含点云）"""
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='坐标系、向量和点云可视化', width=1600, height=1000)
    
    scale_factor = 2.5
    
    # 1. 创建世界坐标系
    world_frame = create_coordinate_frame(world_coor_m_, size=200.0 * scale_factor)
    world_frame.paint_uniform_color([1, 0, 0])  
    
    # 2. 创建局部坐标系
    local_frame = create_coordinate_frame(second_box_m_, size=160 * scale_factor)
    local_frame.paint_uniform_color([0, 0, 1])   
    
    # 3. 创建向量
    world_origin = world_coor_m_[:3, 3]
    local_origin = second_box_m_[:3, 3]
    
    world_vec_cylinder, world_vec_cone = create_vector_arrow(
        local_origin, world_vector, [1, 0, 0], length=300.0 * scale_factor, radius=8)

    
    # 创建平面点云作为参考
    plane_cloud = o3d.io.read_point_cloud('/home/zq/PycharmProjects/DROID-SLAM-cuda11_8/droid_metric/OUTPUT_yolo1/Trajectory/A2_2/pcd_filter_processed.ply')
    


    geometries = [
        world_frame, local_frame, 
        world_vec_cylinder, world_vec_cone,  plane_cloud,
    ]
    # geometries = [
    #     world_frame, 
    #     world_vec_cylinder, world_vec_cone,  plane_cloud,
    # ]

    for geom in geometries:
        vis.add_geometry(geom)
    
    # 9. 设置点云渲染参数
    render_option = vis.get_render_option()
    render_option.point_size = 4.0  # 设置点大小
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深色背景
    
    # 10. 设置视角
    view_control = vis.get_view_control()
    center_point = (local_origin + world_origin) / 2
    view_control.set_front([0, -1, 0.3])
    view_control.set_up([0, 0, 1])
    view_control.set_lookat(center_point)
    view_control.set_zoom(0.6)
    
    # 11. 添加坐标网格
    grid = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8.0)
    grid.paint_uniform_color([0.3, 0.3, 0.3])
    vis.add_geometry(grid)
    
    vis.run()
    vis.destroy_window()
# 创建Open3D内置坐标轴（带箭头）
def create_coordinate_frame1(size, origin):
    """
    使用Open3D内置的create_coordinate_frame创建坐标轴。
    默认X红、Y绿、Z蓝。
    """
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
# 根据状态创建轨迹点的颜色
def get_trajectory_colors(statuses):
    colors = []
    for status in statuses:
        if status == 1.0:
            colors.append([1.0, 0.0, 0.0])  # 红色
        else:
            colors.append([0.0, 0.0, 1.0])  # 蓝色
    return np.array(colors)
# 根据状态创建轨迹线段的颜色（使用起点状态）
def get_line_colors(statuses):
    line_colors = []
    for i in range(len(statuses) - 1):
        status = statuses[i]  # 使用起点状态
        if status == 1.0:
            line_colors.append([1.0, 0.0, 0.0])  # 红色
        else:
            line_colors.append([0.0, 0.0, 1.0])  # 蓝色
    return np.array(line_colors)

def create_line_based_on_projections(line_center, line_direction, start_proj_t, end_proj_t, margin=0.1):
    """
    根据投影点的参数t创建直线段，添加一些边距
    """
    line_direction = line_direction / np.linalg.norm(line_direction)
    
    # 计算直线段的范围，添加边距
    t_min = min(start_proj_t, end_proj_t)
    t_max = max(start_proj_t, end_proj_t)
    
    # 添加边距
    t_range = t_max - t_min
    t_min -= margin * t_range
    t_max += margin * t_range
    
    # 直线的两个端点
    start_point = line_center + t_min * line_direction
    end_point = line_center + t_max * line_direction
    
    # 创建线段
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([start_point, end_point])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    
    return line, t_min, t_max

def visualize_results(trajectory_points=None, translated_trajectory=None):
    """
    使用Open3D和Matplotlib可视化结果
    
    新增参数:
    trajectory_points: 原始轨迹点
    translated_trajectory: 平移后的轨迹点
    """
    
    # ====== 新增部分：轨迹可视化 ======
    geometries = []
    
    if trajectory_points is not None:
        # 创建原始轨迹点云
        traj_pcd = o3d.geometry.PointCloud()
        traj_pcd.points = o3d.utility.Vector3dVector(trajectory_points)
        traj_pcd.paint_uniform_color([1, 1, 0])  # 黄色：原始轨迹
        geometries.append(traj_pcd)
        
        # 可选：创建轨迹线
        if len(trajectory_points) > 1:
            trajectory_lines = []
            for i in range(len(trajectory_points) - 1):
                trajectory_lines.append([i, i + 1])
            
            trajectory_line_set = o3d.geometry.LineSet()
            trajectory_line_set.points = o3d.utility.Vector3dVector(trajectory_points)
            trajectory_line_set.lines = o3d.utility.Vector2iVector(trajectory_lines)
            trajectory_line_set.paint_uniform_color([1, 0.8, 0])  # 橙色
            geometries.append(trajectory_line_set)
    

    if translated_trajectory is not None:
        # 创建平移后的轨迹点云
        translated_pcd = o3d.geometry.PointCloud()
        translated_pcd.points = o3d.utility.Vector3dVector(translated_trajectory)
        translated_pcd.paint_uniform_color([0, 1, 1])  # 青色：平移后的轨迹
        geometries.append(translated_pcd)
        
        # 可选：创建平移后的轨迹线
        if len(translated_trajectory) > 1:
            translated_lines = []
            for i in range(len(translated_trajectory) - 1):
                translated_lines.append([i, i + 1])
            
            translated_line_set = o3d.geometry.LineSet()
            translated_line_set.points = o3d.utility.Vector3dVector(translated_trajectory)
            translated_line_set.lines = o3d.utility.Vector2iVector(translated_lines)
            translated_line_set.paint_uniform_color([0, 0.8, 0.8])  # 浅青色
            geometries.append(translated_line_set)
       
    # ====== 新增部分结束 ======
    
    # 创建坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    geometries.append(coord_frame)
    
    # 设置视角
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D点云直线拟合与轨迹平移", width=1200, height=800)
    
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # 深色背景
    render_option.point_size = 3.0
    
    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    
    vis.run()
    vis.destroy_window()

# 为一组数据创建几何体（支持显示开关）
def create_geometries_for_group(pcd1,pcd2, trajectory,trajectory2, show_pcd=True, show_tra=True, group_name="Group"):
    geometries = []
    geometries.append(pcd1)
    geometries.append(pcd2)

    # 创建轨迹点对象（根据状态着色）——仅当show_tra为True时
    if show_tra and len(trajectory) > 0:
        traj_points = o3d.geometry.PointCloud()
        traj_points.points = o3d.utility.Vector3dVector(trajectory)
        geometries.append(traj_points)

    # 创建轨迹线对象（根据状态着色每个线段）——仅当show_tra为True时
    if show_tra and len(trajectory) >= 2:
        lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        geometries.append(line_set)

    # 创建轨迹点对象（根据状态着色）——仅当show_tra为True时
    if show_tra and len(trajectory2) > 0:
        traj_points2 = o3d.geometry.PointCloud()
        traj_points2.points = o3d.utility.Vector3dVector(trajectory2)
        geometries.append(traj_points2)

    # 创建轨迹线对象（根据状态着色每个线段）——仅当show_tra为True时
    if show_tra and len(trajectory2) >= 2:
        lines2 = [[i, i + 1] for i in range(len(trajectory2) - 1)]
        line_set2 = o3d.geometry.LineSet()
        line_set2.points = o3d.utility.Vector3dVector(trajectory2)
        line_set2.lines = o3d.utility.Vector2iVector(lines2)
        geometries.append(line_set2)

    # 添加坐标轴（在原点，使用Open3D内置形式）——始终添加，作为参考
    origin = [0, 0, 0]
    coord_frame = create_coordinate_frame1(size=50, origin=origin)
    geometries.append(coord_frame)

    # 可选：为轨迹添加一个偏移坐标轴（例如放在轨迹起点）——仅当show_tra为True且有轨迹时添加
    if show_tra and len(trajectory) > 0:  # 修改：添加 show_tra 条件
        traj_origin = trajectory[0].tolist()  # 轨迹起点
        traj_frame = create_coordinate_frame1(size=50, origin=traj_origin)
        geometries.append(traj_frame)
    return geometries

def comparison(ori_tra_path, pro_tra):
    a_trajectory_ = []
    with open(ori_tra_path, 'r') as f_in:
        for line in f_in:
            data = line.strip().split(',')
            if len(data) >= 7:
                pos_direction = np.array([
                    float(data[0]), float(data[1]), float(data[2])
                ])
                a_trajectory_.append(pos_direction)
        
    b_trajectory_ = []
    with open(pro_tra, 'r') as f_in:
        for line in f_in:
            data = line.strip().split(',')
            if len(data) >= 7:
                pos_direction = np.array([
                    float(data[0]), float(data[1]), float(data[2])
                ])
                b_trajectory_.append(pos_direction)

    a = np.array(a_trajectory_)
    b = np.array(b_trajectory_)
    # 7. 计算投影并可视化（添加轨迹和平移轨迹）
    visualize_results(a, b)

def main():
    path = "smooth_tra.txt"
    pathb = "util.txt"
    comparison(path, pathb)



if __name__ == "__main__":
    main()
     