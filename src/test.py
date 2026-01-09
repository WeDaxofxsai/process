import numpy as np
import toppra as ta  # External library for TOPP-RA computations

class TreeNode:
    def __init__(self, x, parent=None, cost=0.0, u=0.0, jerk=0.0):
        self.x = x
        self.parent = parent
        self.cost = cost
        self.u = u
        self.jerk = jerk

def sample_range(x0, q_prime, q_double, N, delta, limits):
    # Use toppra to compute velocity and acceleration bounds without jerk
    # Assume q_prime is waypoints (N+1, n_dof), ss = np.cumsum(delta) with s0=0
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
        cs = instance.controllable_sets[i, :]  # [low, high] for u at gridpoint i, but need x bounds
        # From controllable sets, x bounds are derived from velocity constraints
        # Simplified: compute max min from velocity
        x_lower = 0  # Assume non-negative
        x_upper = min(np.min((vlim[:,1] / q_prime[i,:])**2), np.inf)  # Approximate, adjust per Eq.11
        L.append((x_lower, x_upper))
    return L

def sample_pasf(L_i, V_open_prev, delta_i_minus1, f):
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

def near_parents(x, V_open_prev, num_unvisited, L_i, epsilon):
    delta_x = (L_i[1] - L_i[0]) / num_unvisited
    r = delta_x * epsilon
    near = [node for node in V_open_prev if abs(node.x - x) <= r]
    return near

def find_parent(x, q_prime_i_minus1, q_double_i_minus1, q_triple_i_minus1, V_near, delta_i_minus1, limits):
    costs = {}
    for z in V_near:
        cost = 2 * delta_i_minus1 / (np.sqrt(z.x) + np.sqrt(x))  # Eq.25, s_dot = sqrt(x)
        costs[cost] = z
    sorted_costs = sorted(costs.items())
    for c, z in sorted_costs:
        if valid_node(q_prime_i_minus1, q_double_i_minus1, q_triple_i_minus1, z, x, delta_i_minus1, limits, z.u):
            return z
    return None

def valid_node(q_prime_i_minus1, q_double_i_minus1, q_triple_i_minus1, z, x, delta_i_minus1, limits, prev_u, prev_t=None):
    u = (x - z.x) / (2 * delta_i_minus1)
    # Compute jerk
    if prev_t is None:
        prev_t = delta_i_minus1 / np.sqrt(z.x)  # Approximate time
    jerk = (u - prev_u) / prev_t
    # Compute joint acc and jerk
    s_dot = np.sqrt(x)
    q_dot = q_prime_i_minus1 * s_dot
    q_ddot = q_double_i_minus1 * s_dot**2 + q_prime_i_minus1 * u
    q_dddot = q_triple_i_minus1 * s_dot**3 + 3 * q_double_i_minus1 * s_dot * u + q_prime_i_minus1 * jerk
    # Check bounds
    if np.all(limits['acc'][0] <= q_ddot) and np.all(q_ddot <= limits['acc'][1]) and \
       np.all(limits['jerk'][0] <= q_dddot) and np.all(q_dddot <= limits['jerk'][1]):
        return True
    return False

def connect(parent, child_x, delta_i_minus1, limits):
    u = (child_x - parent.x) / (2 * delta_i_minus1)
    # Compute jerk, assume prev_u = parent.u, prev_t approximate
    prev_t = delta_i_minus1 / np.sqrt(parent.x)
    jerk = (u - parent.u) / prev_t
    cost = parent.cost + 2 * delta_i_minus1 / (np.sqrt(parent.x) + np.sqrt(child_x))
    return TreeNode(child_x, parent, cost, u, jerk)

def solution(T, end_nodes):
    if not end_nodes:
        return None
    min_node = min(end_nodes, key=lambda n: n.cost)
    path = []
    current = min_node
    while current:
        path.append(current.x)
        current = current.parent
    return path[::-1]

def S_TOPP(x0, q_prime, q_double, q_triple, N, delta, limits, f=5, epsilon=10):
    root = TreeNode(x0, cost=0.0, u=0.0, jerk=0.0)
    T = [root]  # List or dict for tree
    V_open = [[root]]  # List of lists, V_open[i]
    L = sample_range(x0, q_prime, q_double, N, delta, limits)
    for i in range(1, N+1):
        V_unvisited_i = sample_pasf(L[i], V_open[i-1], delta[i-1], f)
        V_open_i = []
        for x in V_unvisited_i:
            V_near = near_parents(x, V_open[i-1], len(V_unvisited_i), L[i], epsilon)
            y = find_parent(x, q_prime[i-1], q_double[i-1], q_triple[i-1], V_near, delta[i-1], limits)
            if y:
                node = connect(y, x, delta[i-1], limits)
                V_open_i.append(node)
                T.append(node)
        V_open.append(V_open_i)
        if not V_open_i:
            return None
    return solution(T, V_open[N])