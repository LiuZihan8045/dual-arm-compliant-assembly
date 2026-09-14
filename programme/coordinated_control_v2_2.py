import sim
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ================= 1. 全局配置 =================
STEP_SIZE = 0.05  # 仿真步长

# --- 机械臂 Target 名称 (V-REP中必须存在) ---
TARGET_L_NAME = "/UR10_L_target"
TARGET_R_NAME = "/UR10_R_target"

# --- 阶段 1 & 2 的路径点名称 (V-REP中必须存在) ---
# 左臂点
L_START = "UR10_L_start"
L_P1    = "UR10_L_p1"
L_P2    = "UR10_L_p2"
# 右臂点
R_START = "UR10_R_start"
R_P1    = "UR10_R_p1"
R_P2    = "UR10_R_p2"

# --- 阶段 3 & 4 的物体目标 (在此处定义世界坐标和姿态) ---
# 物体初始姿态 (用户指定)
OBJ_INIT_ORI = [0, -1, 0] 

# 物体目标 T1 (阶段3终点)
OBJ_T1_POS = [0.0, -0.7, 0.7]   # [x, y, z]
OBJ_T1_ORI = [0, -1, -1]        # [alpha, beta, gamma]

# 物体目标 T2 (阶段4终点)
OBJ_T2_POS = [0, -0.7, 0.5]  # [x, y, z]
OBJ_T2_ORI = [0, -1, 1]      # 旋转一下物体

# 阶段时长
DUR_PHASE_1 = 20.0
DUR_PHASE_2 = 5.0
DUR_PHASE_3 = 30.0
DUR_PHASE_4 = 15.0 # 假设为15秒
# ==============================================

def get_matrix_from_handle(client_id, handle):
    """从V-REP句柄获取 4x4 齐次变换矩阵"""
    _, pos = sim.simxGetObjectPosition(client_id, handle, -1, sim.simx_opmode_blocking)
    _, ori = sim.simxGetObjectOrientation(client_id, handle, -1, sim.simx_opmode_blocking)
    return create_matrix(pos, ori)

def create_matrix(pos, euler):
    """根据位置和欧拉角构建 4x4 矩阵"""
    mat = np.eye(4)
    mat[:3, 3] = pos
    # V-REP 默认是 xyz 顺序 (Extrinsic or Intrinsic depends, assuming standard here)
    r = R.from_euler('xyz', euler, degrees=False)
    mat[:3, :3] = r.as_matrix()
    return mat

def decompose_matrix(mat):
    """将 4x4 矩阵分解为 位置 和 欧拉角"""
    pos = mat[:3, 3]
    r = R.from_matrix(mat[:3, :3])
    euler = r.as_euler('xyz', degrees=False)
    return pos, euler

def set_target_pose(client_id, handle, mat):
    """将矩阵应用到 V-REP 对象"""
    pos, ori = decompose_matrix(mat)
    sim.simxSetObjectPosition(client_id, handle, -1, pos.tolist(), sim.simx_opmode_oneshot)
    sim.simxSetObjectOrientation(client_id, handle, -1, ori.tolist(), sim.simx_opmode_oneshot)

def interpolate_pose(start_mat, end_mat, t):
    """
    对两个矩阵进行插值 (位置线性插值 + 姿态球面插值/欧拉角插值)
    这里为了平滑，使用分解插值
    """
    pos_s, ori_s = decompose_matrix(start_mat)
    pos_e, ori_e = decompose_matrix(end_mat)
    
    # 位置插值
    curr_pos = (1 - t) * pos_s + t * pos_e
    
    # 姿态插值 (简单的线性插值对于小角度足够，严谨应使用Slerp)
    # 这里使用简单的欧拉角插值，如需更平滑请改用 Slerp
    curr_ori = (1 - t) * ori_s + t * ori_e
    
    return create_matrix(curr_pos, curr_ori)

def run_independent_phase(client_id, h_L, h_R, mat_L_start, mat_L_end, mat_R_start, mat_R_end, duration, phase_name):
    """
    执行独立运动阶段 (左右臂各自插值)
    """
    steps = int(duration / STEP_SIZE)
    print(f"--- 开始 {phase_name} (时长 {duration}s) ---")
    
    for i in range(steps + 1):
        t = i / steps
        
        # 左臂计算
        curr_L = interpolate_pose(mat_L_start, mat_L_end, t)
        set_target_pose(client_id, h_L, curr_L)
        
        # 右臂计算
        curr_R = interpolate_pose(mat_R_start, mat_R_end, t)
        set_target_pose(client_id, h_R, curr_R)
        
        sim.simxSynchronousTrigger(client_id)
        
    print(f"{phase_name} 完成。")

def run_collaborative_phase(client_id, h_L, h_R, obj_start_mat, obj_end_mat, offset_L, offset_R, duration, phase_name):
    """
    执行协同搬运阶段 (插值物体，反算手臂)
    """
    steps = int(duration / STEP_SIZE)
    print(f"--- 开始 {phase_name} (协同搬运, 时长 {duration}s) ---")
    
    for i in range(steps + 1):
        t = i / steps
        
        # 1. 计算当前时刻 物体 的位姿
        curr_obj_mat = interpolate_pose(obj_start_mat, obj_end_mat, t)
        
        # 2. 利用相对变换矩阵，算出左右臂应该在的位置
        # T_arm = T_obj * T_offset
        target_L_mat = np.dot(curr_obj_mat, offset_L)
        target_R_mat = np.dot(curr_obj_mat, offset_R)
        
        # 3. 发送给 V-REP
        set_target_pose(client_id, h_L, target_L_mat)
        set_target_pose(client_id, h_R, target_R_mat)
        
        sim.simxSynchronousTrigger(client_id)
        
    print(f"{phase_name} 完成。")
    return obj_end_mat # 返回物体最终位置作为下一阶段起点

def main():
    # 1. 连接
    sim.simxFinish(-1)
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if client_id == -1:
        print("连接失败")
        return
    
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)
    
    try:
        # 2. 获取句柄
        def get_h(name):
            _, h = sim.simxGetObjectHandle(client_id, name, sim.simx_opmode_blocking)
            return h
            
        h_L_target = get_h(TARGET_L_NAME)
        h_R_target = get_h(TARGET_R_NAME)
        
        # 读取路径点矩阵
        mat_L_start = get_matrix_from_handle(client_id, get_h(L_START))
        mat_L_p1    = get_matrix_from_handle(client_id, get_h(L_P1))
        mat_L_p2    = get_matrix_from_handle(client_id, get_h(L_P2))
        
        mat_R_start = get_matrix_from_handle(client_id, get_h(R_START))
        mat_R_p1    = get_matrix_from_handle(client_id, get_h(R_P1))
        mat_R_p2    = get_matrix_from_handle(client_id, get_h(R_P2))
        
        # ================= 第一阶段: Start -> P1 =================
        run_independent_phase(client_id, h_L_target, h_R_target, 
                              mat_L_start, mat_L_p1, 
                              mat_R_start, mat_R_p1, 
                              DUR_PHASE_1, "阶段1 (Start->P1)")
                              
        # ================= 第二阶段: P1 -> P2 =================
        run_independent_phase(client_id, h_L_target, h_R_target, 
                              mat_L_p1, mat_L_p2, 
                              mat_R_p1, mat_R_p2, 
                              DUR_PHASE_2, "阶段2 (P1->P2)")
                              
        # ================= 协同准备工作 =================
        print("计算协同偏移量...")
        # 此时机械臂应该正好在 P2 位置
        # 1. 计算物体初始中心位置 (两臂中心)
        pos_L_p2, _ = decompose_matrix(mat_L_p2)
        pos_R_p2, _ = decompose_matrix(mat_R_p2)
        obj_init_pos = (pos_L_p2 + pos_R_p2) / 2.0
        
        # 2. 构建物体初始矩阵 (位置=中心, 姿态=用户定义)
        mat_obj_start = create_matrix(obj_init_pos, OBJ_INIT_ORI)
        
        # 3. 计算“抓取偏移量” (Offset = Obj_inv * Arm)
        # 这样无论物体怎么动，Arm = Obj_new * Offset 都能保持相对静止
        mat_obj_inv = np.linalg.inv(mat_obj_start)
        offset_L = np.dot(mat_obj_inv, mat_L_p2)
        offset_R = np.dot(mat_obj_inv, mat_R_p2)
        
        # 准备 T1 和 T2 矩阵
        mat_obj_t1 = create_matrix(OBJ_T1_POS, OBJ_T1_ORI)
        mat_obj_t2 = create_matrix(OBJ_T2_POS, OBJ_T2_ORI)
        
        # ================= 第三阶段: 物体 Start -> T1 =================
        run_collaborative_phase(client_id, h_L_target, h_R_target,
                                mat_obj_start, mat_obj_t1,
                                offset_L, offset_R,
                                DUR_PHASE_3, "阶段3 (协同->T1)")
                                
        # ================= 第四阶段: 物体 T1 -> T2 =================
        run_collaborative_phase(client_id, h_L_target, h_R_target,
                                mat_obj_t1, mat_obj_t2,
                                offset_L, offset_R,
                                DUR_PHASE_4, "阶段4 (协同->T2)")

        print("所有任务完成。")
        for _ in range(50): sim.simxSynchronousTrigger(client_id)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)
        sim.simxFinish(client_id)

if __name__ == "__main__":
    main()