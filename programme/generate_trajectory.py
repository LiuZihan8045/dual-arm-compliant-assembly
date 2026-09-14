import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os

# ================= 配置区域 =================
PART_GOAL_POS = np.array([0.0, -0.775, 0.220])
PART_INIT_ORI_EULER = np.array([0, 0, 0])
PART_GOAL_ORI = np.array([0, 0, 0])

# 假设起始位置 (你需要根据实际情况修改，或者先运行一次 sim 获取)
# 这里为了生成独立文件，我使用你之前代码中的逻辑起点
# 注意：实际使用时，最好是先读取当前 V-REP 中的物体位置作为起点
# 这里暂定一个理论起点，请根据实际 V-REP 场景调整
PART_START_POS = np.array([0.0, -0.4, 0.5]) # 示例起点

DURATION = 20.0
STEP_SIZE = 0.05
TOTAL_STEPS = int(DURATION / STEP_SIZE)

OUTPUT_FILE = "ideal_trajectory.npy"

def generate_trajectory():
    print(f"正在生成 {TOTAL_STEPS} 步的理想轨迹...")
    
    # 初始姿态矩阵
    r_start = R.from_euler('xyz', PART_INIT_ORI_EULER)
    r_goal = R.from_euler('xyz', PART_GOAL_ORI)
    
    # 构建关键帧旋转
    key_rots = R.from_matrix([r_start.as_matrix(), r_goal.as_matrix()])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    
    trajectory_data = []

    for step in range(TOTAL_STEPS + 1):
        t = step / TOTAL_STEPS
        
        # 1. 位置线性插值
        pos = (1 - t) * PART_START_POS + t * PART_GOAL_POS
        
        # 2. 姿态球面插值
        rot = slerp([t]).as_matrix()[0]
        quat = R.from_matrix(rot).as_quat() # [x, y, z, w]
        
        # 保存格式: [step, x, y, z, qx, qy, qz, qw]
        data_point = np.concatenate(([step], pos, quat))
        trajectory_data.append(data_point)

    # 保存文件
    np.save(OUTPUT_FILE, np.array(trajectory_data))
    print(f"轨迹已保存至 {OUTPUT_FILE}")
    print(f"数据形状: {np.array(trajectory_data).shape}")
    print("列定义: [step, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]")

if __name__ == "__main__":
    # 注意：如果想起点精确对应 V-REP，建议在主控制脚本里实时生成，
    # 但为了满足你"先跑一下记录"的要求，这里是一个独立生成器。
    # 更好的做法是在主控制脚本里，第一阶段先跑一遍生成，第二阶段再跑跟踪。
    # 下面的跟踪脚本会采用"实时生成+实时记录"的方式，更加灵活。
    generate_trajectory()