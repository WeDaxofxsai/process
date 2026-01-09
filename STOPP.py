import math
import toppra as ta
import toppra.constraint as constraint
import numpy as np
import toppra.algorithm as algo
"""
S-TOPP旨在解决时间最有优路径参数化问题
"""


class STOPP():
    def __init__(self, file, vlim, alim, epsilon):
        self.s = None
        self,x = None
        self.file_route = file
        self.epsilon = epsilon
        self.vlim = vlim
        self.alim = alim

    def sample_range(self, vlim, alim):
        """
        基于速度和加速度约束计算每个阶段的x范围
        忽略加加速度约束（因为会导致非凸问题）
        使用TOPP-RA方法计算可达集边界
        return:L = [[x_min^1, x_max^1], [x_min^2, x_max^2], ..., [x_min^N, x_max^N]]
        """
        joint_trajectory = []
        with open(self.file_route, 'r') as f:
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

        path = ta.SplineInterpolator(path_param, joint_waypoints)
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(alim)
        instance = algo.TOPPRA([pc_vel, pc_acc], path)
        sd_start = 0.0
        sd_end = 0.0

        try:
            _, _, _, K = instance.compute_parameterization(
                sd_start, sd_end, return_data=True
            )
            s_dot_min = K[:, 0]  # 最小速度
            s_dot_max = K[:, 1]  # 最大速度
            # if hasattr(instance, 'gridpoints'):
            #     ss = instance.gridpoints
            # else:
            #     # 从problem_data获取
            #     problem_data = instance.problem_data
            #     if 'gridpoints' in problem_data:
            #         ss = problem_data['gridpoints']
            #     else:
            #         # 默认创建
            #         N = len(sd_vec) - 1
            #         ss = np.linspace(0, 1, N+1)
            # x_upper = s_dot_max**2
            # return ss, s_dot_min, s_dot_max, x_upper, sd_vec
            return np.array([s_dot_min**2, s_dot_max**2])
            
        except Exception as e:
            print(f"计算失败: {e}")
            return None

    def sample_pasf(L_i, V_open_prev, delta_i_minus1, f):
        """
        使用历史的路径加速的去预测这一步可能的x
        """
        samples = []
        # PASF: use historical u to predict x_i
        for node in V_open_prev:
            u_prev = node.u  # Assume u_{i-2} is stored as node.u
            x_pred = node.x + 2 * delta_i_minus1 * u_prev
            if L_i[0] <= x_pred <= L_i[1]:
                samples.append(x_pred)
        # Add f uniform samples
        uniform = np.linspace(L_i[0], L_i[1], f)
        samples.extend(uniform)
        return list(set(samples))  # Remove duplicates

    def NearParents(self, x, V_open_prev, L_i, num_unvisited):
        """
        input:
            x: 当前采样点
            V_open_prev: 上一阶段开放节点集合
            L_i: 该点的速度上下限
        过程:
            1. 计算搜索半径 r = Δ_x·ε
            2. 在V_open_prev中寻找与x距离小于r的节点
        return:候选父节点集合
        """
        delta_x = (L_i[1] - L_i[0]) / num_unvisited
        r = delta_x * self.epsilon
        near = [node for node in V_open_prev if abs(node.x - x) <= r]
        return near

    def find_parent(self, x, q_, q__, q___, V_near):
        """         
        input:
           x: 当前采样点
           q', q'', q''': 路径导数
           V_near: 候选父节点集合
        过程:
            1. 计算每个候选父节点到x的成本(时间)
            2. 按成本升序排序
            3. 从最小成本开始,检查是否满足约束
            4. 一旦找到满足约束的父节点,立即return 
            5. 如果所有都不满足,return None
        return:最佳父节点或None
        """
        def find_parent(x, q_prime_i_minus1, q_double_i_minus1, q_triple_i_minus1, V_near, delta_i_minus1, limits):
            costs = {}
            for z in V_near:
                cost = 2 * delta_i_minus1 / (np.sqrt(z.x) + np.sqrt(x))  # s_dot = sqrt(x)
                costs[cost] = z
            sorted_costs = sorted(costs.items())
            for c, z in sorted_costs:
                if self.valid_node(q_prime_i_minus1, q_double_i_minus1, q_triple_i_minus1, z, x, delta_i_minus1, limits, z.u):
                    return z
            return None

    def valid_node(self, ):
        pass

    def Connect(self, parent, child):
        """     
        建立父子节点连接
        计算控制input u = (x_child - x_parent)/(2Δs)
        计算时间成本 Δt = 2Δs/(√x_parent + √x_child)
        更新孩子的累积成本 = 父节点成本 + Δt
        存储连接信息
        """
        pass

    def Solution(self,T):
        """     
        从树的最后一个阶段(终点)开始
        找到成本最小的节点
        沿着父节点指针回溯到起点
        return 完整的轨迹序列
        """
        pass

    def process(self):
        pass



class TreeNode:
    def __init__(self, x, parent=None, cost=0.0, u=0.0, jerk=0.0):
        self.x = x
        self.parent = parent
        self.cost = cost
        self.u = u
        self.jerk = jerk

