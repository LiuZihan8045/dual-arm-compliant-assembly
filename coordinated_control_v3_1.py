import sim
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# ================= 配置区域 =================
# V-REP 中的对象名称
NAME_L_P2 = "UR10_L_p2"       # 左臂抓取点
NAME_R_P2 = "UR10_R_p2"       # 右臂抓取点
NAME_PART = "Part1_Reference" # 我们刚才创建的物体Dummy
NAME_L_TARGET = "/UR10_L_target" # 左臂IK目标
NAME_R_TARGET = "/UR10_R_target" # 右臂IK目标

# 仿真参数
STEP_SIZE = 0.05
DURATION = 20.0  # 搬运总时长

# --- Part1 的目标位姿 (用户自定义) ---
# 目标位置 (世界坐标)
PART_GOAL_POS = [0.0, -0.775, 0.220] 
# 目标姿态 (欧拉角 [alpha, beta, gamma] / 弧度)
# 这里假设你想让物体旋转90度或者保持不变，根据需求修改
PART_GOAL_ORI = [0, 0, 0] 

# Part1 的初始姿态 (用户指定)
PART_INIT_ORI_EULER = [0, 0, 0]
# ===========================================

def connect_vrep():
    print("连接 V-REP...")
    sim.simxFinish(-1)
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if client_id == -1:
        print("连接失败")
        sys.exit()
    return client_id

def create_matrix(pos, quat):
    """pos: [x,y,z], quat: [x,y,z,w]"""
    mat = np.eye(4)
    mat[:3, 3] = pos
    r = R.from_quat(quat)
    mat[:3, :3] = r.as_matrix()
    return mat

def get_matrix_from_handle(client_id, handle):
    _, pos = sim.simxGetObjectPosition(client_id, handle, -1, sim.simx_opmode_blocking)
    _, quat = sim.simxGetObjectQuaternion(client_id, handle, -1, sim.simx_opmode_blocking)
    return create_matrix(pos, quat)

def set_target_pose(client_id, handle, mat):
    """将矩阵分解为位置和四元数发送给V-REP"""
    pos = mat[:3, 3]
    r = R.from_matrix(mat[:3, :3])
    quat = r.as_quat() # [x, y, z, w]
    
    sim.simxSetObjectPosition(client_id, handle, -1, pos.tolist(), sim.simx_opmode_oneshot)
    sim.simxSetObjectQuaternion(client_id, handle, -1, quat.tolist(), sim.simx_opmode_oneshot)

def interpolate_matrix(start_mat, end_mat, t):
    """
    SLERP 插值核心函数
    """
    # 1. 位置线性插值
    pos_s = start_mat[:3, 3]
    pos_e = end_mat[:3, 3]
    curr_pos = (1 - t) * pos_s + t * pos_e
    
    # 2. 姿态球面插值 (SLERP)
    rot_s = R.from_matrix(start_mat[:3, :3])
    rot_e = R.from_matrix(end_mat[:3, :3])
    
    key_rots = R.from_matrix([start_mat[:3, :3], end_mat[:3, :3]])
    slerp = Slerp([0, 1], key_rots)
    curr_rot = slerp([t])
    
    # 3. 合成矩阵
    mat = np.eye(4)
    mat[:3, 3] = curr_pos
    mat[:3, :3] = curr_rot.as_matrix()[0]
    return mat

def main():
    client_id = connect_vrep()
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)
    
    try:
        # 1. 获取所有句柄
        def get_h(name):
            res, h = sim.simxGetObjectHandle(client_id, name, sim.simx_opmode_blocking)
            if res != 0: print(f"Err: 找不到 {name}")
            return h

        h_L_p2 = get_h(NAME_L_P2)
        h_R_p2 = get_h(NAME_R_P2)
        h_Part = get_h(NAME_PART)
        h_L_target = get_h(NAME_L_TARGET)
        h_R_target = get_h(NAME_R_TARGET)
        
        # 2. 计算物体的【实际初始位置】 (基于两手位置计算)
        print("计算初始状态...")
        mat_L_p2 = get_matrix_from_handle(client_id, h_L_p2)
        mat_R_p2 = get_matrix_from_handle(client_id, h_R_p2)
        
        pos_L = mat_L_p2[:3, 3]
        pos_R = mat_R_p2[:3, 3]
        
        # 中点作为物体初始位置
        part_init_pos = (pos_L + pos_R) / 2.0
        
        # 构建物体初始矩阵 (位置=计算出的中点, 姿态=用户指定的[0,-1,0])
        r_init = R.from_euler('xyz', PART_INIT_ORI_EULER, degrees=False)
        mat_part_start = create_matrix(part_init_pos, r_init.as_quat())
        
        # 3. 将 V-REP 中的物体瞬间移动到这个计算出的初始位置 (对齐)
        # 这一步保证了逻辑闭环：物体此刻就在两手中间
        set_target_pose(client_id, h_Part, mat_part_start)
        sim.simxSynchronousTrigger(client_id) 
        
        # 4. 计算“抓取偏移量” (Offset)
        # 这是一个相对变换矩阵：描述“手在物体的什么方位”
        # Offset = Inv(Part) * Hand
        mat_part_inv = np.linalg.inv(mat_part_start)
        offset_L = np.dot(mat_part_inv, mat_L_p2)
        offset_R = np.dot(mat_part_inv, mat_R_p2)
        
        # 5. 构建物体目标矩阵
        r_goal = R.from_euler('xyz', PART_GOAL_ORI, degrees=False)
        mat_part_goal = create_matrix(PART_GOAL_POS, r_goal.as_quat())
        
        print(f"=== 开始协同搬运 (SLERP) ===")
        print(f"初始位置: {part_init_pos}")
        print(f"目标位置: {PART_GOAL_POS}")
        
        steps = int(DURATION / STEP_SIZE)
        
        for i in range(steps + 1):
            t = i / steps
            
            # A. 插值计算物体当前的矩阵
            curr_part_mat = interpolate_matrix(mat_part_start, mat_part_goal, t)
            
            # B. 移动物体 (用于视觉显示)
            set_target_pose(client_id, h_Part, curr_part_mat)
            
            # C. 反算机械臂目标 (Part * Offset)
            target_L_mat = np.dot(curr_part_mat, offset_L)
            target_R_mat = np.dot(curr_part_mat, offset_R)
            
            # D. 移动机械臂目标
            set_target_pose(client_id, h_L_target, target_L_mat)
            set_target_pose(client_id, h_R_target, target_R_mat)
            
            sim.simxSynchronousTrigger(client_id)
            
        print("搬运完成")
        for _ in range(50): sim.simxSynchronousTrigger(client_id)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)
        sim.simxFinish(client_id)

if __name__ == "__main__":
    main()