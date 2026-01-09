import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
处理奇异关节，使得关节轨迹平滑
"""



# 读取数据
data = pd.read_csv('toppra_tra.txt', header=None)

# 假设有6个关节，可以根据实际数据调整
num_joints = data.shape[1] if data.shape[1] <= 6 else 6
variable_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6'][:num_joints]

# 将数据转换为numpy数组以便处理
original_data = data.iloc[:, :num_joints].values.copy()
processed_data = original_data.copy()

# 定义角度转换（30度 = π/6 弧度）
threshold_rad = 30 * np.pi / 180  # 30度转换为弧度
two_pi = 2 * np.pi

# 用于存储所有突变区域
all_mutations = []

# 1. 对每个关节进行循环，找出突变区域
for joint_idx in range(num_joints):
    print(f"\n处理关节 {joint_idx+1} ({variable_names[joint_idx]})...")
    
    joint_data = original_data[:, joint_idx]
    mutations = []
    current_mutation = None
    i = 0
    
    while i < len(joint_data) - 1:
        diff = abs(joint_data[i+1] - joint_data[i])
        
        if diff > threshold_rad:
            # 如果当前没有开启存值模式，则开启
            if current_mutation is None:
                current_mutation = [i]
                # 也存入i+1，因为突变发生在i和i+1之间
                if i+1 not in current_mutation:
                    current_mutation.append(i+1)
            else:
                # 已经在存值模式中，将i+1添加到列表
                if i+1 not in current_mutation:
                    current_mutation.append(i+1)
            
            # 继续检查下一个点
            i += 1
        else:
            # 如果突变结束且存值模式已开启
            if current_mutation is not None:
                if len(current_mutation) > 1:
                    mutations.append(current_mutation)
                current_mutation = None
            i += 1
    
    # 检查最后一个突变区域是否结束
    if current_mutation is not None and len(current_mutation) > 1:
        mutations.append(current_mutation)
    
    # 存储这个关节的突变区域
    all_mutations.append(mutations)
    
    print(f"  找到 {len(mutations)} 个突变区域")


# 2. 对每个突变区域进行处理
for joint_idx in range(num_joints):
    print(f"\n处理关节 {joint_idx+1} 的突变区域...")
    joint_mutations = all_mutations[joint_idx]
    
    if not joint_mutations:
        print("  没有突变区域需要处理")
        continue
    
    for mutation_idx, mutation_list in enumerate(joint_mutations):
        print(f"  处理突变区域 {mutation_idx+1}: {mutation_list}")
        if len(mutation_list) < 1:
            continue
        # 获取突变区域的起始和结束索引
        start_idx = mutation_list[0]
        end_idx = mutation_list[-1]
        
        # 获取起始和结束值
        start_val = processed_data[start_idx, joint_idx]
        end_val = processed_data[end_idx, joint_idx]
        
        # 判断差值是否小于2π
        diff = abs(end_val - start_val)
        
        if diff >= two_pi/2:
            # 计算需要调整的值（+或-2π）
            adjustment = 0
            if start_val < end_val:
                # 可能需要减去2π
                processed_data[0:start_idx+1, joint_idx] += two_pi
            else:
                processed_data[0:start_idx+1, joint_idx] -= two_pi
            
            # 重新获取调整后的起始值
            start_val = processed_data[start_idx, joint_idx]
            end_val = processed_data[end_idx, joint_idx]
        
        # 对突变区域内的点进行线性插值
        if len(mutation_list) > 2:
            # 跳过第一个和最后一个点（已确定）
            for i in range(1, len(mutation_list)-1):
                idx = mutation_list[i]
                # 线性插值
                t = (idx - start_idx) / (end_idx - start_idx)
                interpolated_val = start_val + (end_val - start_val) * t
                processed_data[idx, joint_idx] = interpolated_val
                print(f"    索引 {idx}: {original_data[idx, joint_idx]:.4f} -> {interpolated_val:.4f}")

# 3. 创建处理前后的对比图
fig, axes = plt.subplots(num_joints, 2, figsize=(15, 3*num_joints))
fig.suptitle('关节角度处理前后对比（弧度）', fontsize=16, fontweight='bold')

for joint_idx in range(num_joints):
    # 原始数据
    ax1 = axes[joint_idx, 0]
    ax1.plot(original_data[:, joint_idx], marker='o', markersize=3, linewidth=1, label='原始')
    ax1.set_title(f'{variable_names[joint_idx]} - 原始数据')
    ax1.set_xlabel('样本序号')
    ax1.set_ylabel('弧度')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-2*np.pi, 2*np.pi])
    # 标记突变区域
    for mutation in all_mutations[joint_idx]:
        if len(mutation) > 0:
            start_idx = mutation[0]
            end_idx = mutation[-1]
            ax1.axvspan(start_idx, end_idx, alpha=0.3, color='red')
    
    # 处理后的数据
    ax2 = axes[joint_idx, 1]
    ax2.plot(processed_data[:, joint_idx], marker='o', markersize=3, linewidth=1, label='处理后', color='orange')
    ax2.set_title(f'{variable_names[joint_idx]} - 处理后数据')
    ax2.set_xlabel('样本序号')
    ax2.set_ylabel('弧度')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([-2*np.pi, 2*np.pi])
plt.tight_layout()
plt.show()



# 5. 保存处理后的数据
output_file = 'processed_joints.txt'
np.savetxt(output_file, processed_data, fmt='%.6f')