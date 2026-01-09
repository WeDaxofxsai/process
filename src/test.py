import math
import toppra as ta
import toppra.constraint as constraint
import numpy as np
import toppra.algorithm as algo
import matplotlib.pyplot as plt

def get_x_upper_bound_via_toppra_api(file, vlim, alim):
    """使用TOPPRA API直接获取x上界"""
    

    joint_trajectory = []
    with open(file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            values = line.split(',')
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
    joint_waypoints = np.array(joint_trajectory)
    path_param = np.linspace(0, 1, len(joint_waypoints))

    path = ta.SplineInterpolator(
        path_param,
        joint_waypoints
    )
    
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim)
    instance = algo.TOPPRA([pc_vel, pc_acc], path)
    sd_start = 0.0
    sd_end = 0.0
    
    try:
        sdd_vec, sd_vec, v_vec, K = instance.compute_parameterization(
            sd_start, sd_end, return_data=True
        )
        s_dot_min = K[:, 0]  # 最小速度
        s_dot_max = K[:, 1]  # 最大速度
        
        if hasattr(instance, 'gridpoints'):
            ss = instance.gridpoints
        else:
            # 从problem_data获取
            problem_data = instance.problem_data
            if 'gridpoints' in problem_data:
                ss = problem_data['gridpoints']
            else:
                # 默认创建
                N = len(sd_vec) - 1
                ss = np.linspace(0, 1, N+1)
        
        x_upper = s_dot_max**2
        return ss, s_dot_min, s_dot_max, x_upper, sd_vec
        
    except Exception as e:
        print(f"计算失败: {e}")
        return None

def visualize_results(ss, s_dot_min, s_dot_max, x_upper, sd_vec):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 速度可行区间
    ax = axes[0, 0]
    ax.fill_between(ss, s_dot_min, s_dot_max, alpha=0.3, color='blue', label='Feasible region')
    ax.plot(ss, s_dot_min, 'r--', label='Minimum speed')
    ax.plot(ss, s_dot_max, 'g-', label='Maximum speed')
    if sd_vec is not None:
        ax.plot(ss, sd_vec**2, 'b-', linewidth=2, label='Actual speed')
    ax.set_xlabel('Path parameter $s$')
    ax.set_ylabel('Path velocity $\dot{s}$')
    ax.set_title('Velocity Feasible Region')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":

    vlim = np.array([[-1.74, 1.74], [-1.74, 1.74], [-1.74, 1.74], [-1.74, 1.74], [-1.74, 1.74], [-1.74, 1.74]])  # 关节速度限制
    alim = np.array([[-1.74, 1.74], [-1.74, 1.74], [-1.74, 1.74], [-1.74, 1.74], [-1.74, 1.74], [-1.74, 1.74]])  # 关节加速度限制
    result = get_x_upper_bound_via_toppra_api("./source/test_joint.txt", vlim, alim)
    
    if result is not None:
        ss, s_dot_min, s_dot_max, x_upper, sd_vec = result
        
        # 打印基本信息
        print("ss ", len(ss))
        print("s_dot_min ", len(s_dot_min))
        print("s_dot_max ", len(s_dot_max))
        print("x_upper ", len(x_upper))
        print("sd_vec ", len(sd_vec))
        print(f"网格点数量: {len(ss)}")
        print(f"速度可行区间形状: {s_dot_min.shape}")
        print(f"x上界形状: {x_upper.shape}")
        
        # 可视化
        visualize_results(ss, s_dot_min, s_dot_max, x_upper, sd_vec)
        print("数据已保存到 velocity_feasible_region.npz")
    else:
        print("无法获取速度可行区间")