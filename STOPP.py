import toppra as ta
import numpy as np
"""
S-TOPP旨在解决时间最有优路径参数化问题
"""


class STOPP():
    def __init__(self):
        self.s = None
        self,x = None

    def SampleRange(self, x_path, q_, q__, q___):
        """
        基于速度和加速度约束计算每个阶段的x范围
        忽略加加速度约束（因为会导致非凸问题）
        使用TOPP-RA方法计算可达集边界
        return:L = [[x_min^1, x_max^1], [x_min^2, x_max^2], ..., [x_min^N, x_max^N]]
        """
        pass

    def sample_range(self,x0, q_prime, q_double, N, delta, limits):
        """
        input:当前阶段的采样范围 L_i = [x_min^i, x_max^i]
        过程:
            1. 对于上一阶段的每个节点,假设其加速度不变,预测当前阶段的值
            2. 添加f个冗余样本(均匀采样)
            3. 限制在L_i范围内
        return:当前阶段的采样点集合
        """
        ss = np.zeros(N+1)
        ss[1:] = np.cumsum(delta)
        path = ta.SplineInterpolator(ss, q_prime)  # Geometric path
        vlim = limits['vel']  # [q_min, q_max] shape (n_dof, 2)
        alim = limits['acc']  # [q_ddot_min, q_ddot_max] shape (n_dof, 2)
        pc_vel = ta.constraint.JointVelocityConstraint(vlim)
        pc_acc = ta.constraint.JointAccelerationConstraint(alim)
        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc], path, gridpoints_min= N+1)
        _ = instance.compute_trajectory(0, 0)  # Compute to get controllable sets
        # Extract bounds for x_i = s_dot^2
        L = []
        for i in range(N+1):
            
            """
            cs = instance.controllable_sets[i, :]  # [low, high] for u at gridpoint i, but need x bounds
            x_upper = min(np.min((vlim[:,1] / q_prime[i,:])**2), np.inf)
            """
            x_lower = 0
            x_upper = 0
            L.append((x_lower, x_upper))
        return L

    def NearParents(self, x, V_open_prev, V_unvisited):
        """
        input:
            x: 当前采样点
            V_open_prev: 上一阶段开放节点集合
            V_unvisited: 当前阶段所有采样点(用于计算Δ_x)
        过程:
            1. 计算搜索半径 r = Δ_x·ε
            2. 在V_open_prev中寻找与x距离小于r的节点
        return:候选父节点集合
        """
        pass

    def FindParent(self, x, q_, q__, q___, V_near):
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

    def Solution(T):
        """     
        从树的最后一个阶段(终点)开始
        找到成本最小的节点
        沿着父节点指针回溯到起点
        return 完整的轨迹序列
        """
        pass



class TreeNode:
    def __init__(self, x, parent=None, cost=0.0, u=0.0, jerk=0.0):
        self.x = x
        self.parent = parent
        self.cost = cost
        self.u = u
        self.jerk = jerk

