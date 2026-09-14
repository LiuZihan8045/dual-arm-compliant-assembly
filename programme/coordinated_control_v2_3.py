import sim
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp # 引入球面插值

# ================= 1. 全局配置 =================
STEP_SIZE = 0.05  # 仿真步长

# --- 机械臂 Target 名称 (V-REP中必须存在) ---
TARGET_L_NAME = "/UR10_L_target"
TARGET_R_NAME = "/UR10_R_target"

# --- 阶段 1 & 2 的路径点名称 ---
L_START = "UR10_L_start"
L_P1    = "UR10_L_p1"
L_P2    = "UR10_L_p2"
R_START = "UR10_R_start"
R_P1    = "UR10_R_p1"
R_P2    = "UR10_R_p2"

# --- 阶段 3 & 4 的物体目标 ---
OBJ_INIT_ORI = [0, -1, 0] 

# 物体目标 T1
OBJ_T1_POS = [0.0, -0.7, 0.7]  
OBJ_T1_ORI = [0, -1, -1]        

# 物体目标 T2
OBJ_T2_POS = [0.0, -0.7, 0.7]
OBJ_T2_ORI = [0, -1, 1]      

# 阶段时长
DUR_PHASE_1 = 20.0
DUR_PHASE_2 = 5.0
DUR_PHASE_3 = 30.0
DUR_PHASE_4 = 15.0 
# ==============================================

def get_matrix_from_handle(client_id, handle):
    """从V-REP句柄获取 4x4 齐次变换矩阵"""
    _, pos = sim.simxGetObjectPosition(client_id, handle, -1, sim.simx_opmode_blocking)
    # 获取四元数比获取欧拉角更稳定，虽然V-REP旧API习惯用欧拉角，但这里我们尽量避免
    # 为了兼容旧代码逻辑，这里暂且用 Orientation 获取，但在构建矩阵时处理
    _, ori = sim.simxGetObjectOrientation(client_id, handle, -1, sim.simx_opmode_blocking)
    return create_matrix(pos, ori)

def create_matrix(pos, euler_or_quat):
    """根据位置和欧拉角构建 4x4 矩阵"""
    mat = np.eye(4)
    mat[:3, 3] = pos
    # 这里的输入可能是欧拉角
    r = R.from_euler('xyz', euler_or_quat, degrees=False)
    mat[:3, :3] = r.as_matrix()
    return mat

def set_target_pose(client_id, handle, mat):
    """
    [核心修改] 将矩阵应用到 V-REP 对象
    使用 Quaternion (四元数) 替代 Euler Angle，彻底解决 Gimbal Lock
    """
    pos = mat[:3, 3]
    # 从矩阵提取旋转对象
    r = R.from_matrix(mat[:3, :3])
    # 转换为四元数 [x, y, z, w]
    quat = r.as_quat()
    
    # 设置位置
    sim.simxSetObjectPosition(client_id, handle, -1, pos.tolist(), sim.simx_opmode_oneshot)
    # 设置姿态 (使用四元数接口)
    sim.simxSetObjectQuaternion(client_id, handle, -1, quat.tolist(), sim.simx_opmode_oneshot)

def interpolate_pose(start_mat, end_mat, t):
    """
    [核心修改] 对两个矩阵进行插值
    位置：线性插值 (Lerp)
    姿态：球面线性插值 (Slerp) - 比欧拉角插值更平滑且无死锁
    """
    # 1. 提取位置并线性插值
    pos_s = start_mat[:3, 3]
    pos_e = end_mat[:3, 3]
    curr_pos = (1 - t) * pos_s + t * pos_e
    
    # 2. 提取旋转并 Slerp 插值
    rot_s = R.from_matrix(start_mat[:3, :3])
    rot_e = R.from_matrix(end_mat[:3, :3])
    
    #以此构建关键帧 (Times: 0 -> 1, Rots: Start -> End)
    key_rots = R.from_matrix([start_mat[:3, :3], end_mat[:3, :3]])
    slerp = Slerp([0, 1], key_rots)
    
    # 计算当前时刻 t 的旋转
    curr_rot = slerp([t])
    
    # 3. 重新组合成 4x4 矩阵
    mat = np.eye(4)
    mat[:3, 3] = curr_pos
    mat[:3, :3] = curr_rot.as_matrix()[0] # slerp返回的是数组，取第一个
    
    return mat

def decompose_matrix(mat):
    """
    辅助函数：仅用于获取位置 (不再用于提取欧拉角，避免报错)
    """
    pos = mat[:3, 3]
    # 如果还需要欧拉角做打印调试，可以在这里加 try-catch 或者忽略警告
    return pos, None 

def run_independent_phase(client_id, h_L, h_R, mat_L_start, mat_L_end, mat_R_start, mat_R_end, duration, phase_name):
    steps = int(duration / STEP_SIZE)
    print(f"--- 开始 {phase_name} (时长 {duration}s) ---")
    
    for i in range(steps + 1):
        t = i / steps
        
        curr_L = interpolate_pose(mat_L_start, mat_L_end, t)
        set_target_pose(client_id, h_L, curr_L)
        
        curr_R = interpolate_pose(mat_R_start, mat_R_end, t)
        set_target_pose(client_id, h_R, curr_R)
        
        sim.simxSynchronousTrigger(client_id)
        
    print(f"{phase_name} 完成。")

def run_collaborative_phase(client_id, h_L, h_R, obj_start_mat, obj_end_mat, offset_L, offset_R, duration, phase_name):
    steps = int(duration / STEP_SIZE)
    print(f"--- 开始 {phase_name} (协同搬运, 时长 {duration}s) ---")
    
    for i in range(steps + 1):
        t = i / steps
        
        # 1. 计算当前时刻 物体 的位姿 (使用 Slerp)
        curr_obj_mat = interpolate_pose(obj_start_mat, obj_end_mat, t)
        
        # 2. 矩阵乘法算出两臂目标
        target_L_mat = np.dot(curr_obj_mat, offset_L)
        target_R_mat = np.dot(curr_obj_mat, offset_R)
        
        # 3. 发送四元数给 V-REP (不会再报 Gimbal Lock)
        set_target_pose(client_id, h_L, target_L_mat)
        set_target_pose(client_id, h_R, target_R_mat)
        
        sim.simxSynchronousTrigger(client_id)
        
    print(f"{phase_name} 完成。")
    return obj_end_mat 

def main():
    sim.simxFinish(-1)
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if client_id == -1:
        print("连接失败")
        return
    
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)
    
    try:
        def get_h(name):
            _, h = sim.simxGetObjectHandle(client_id, name, sim.simx_opmode_blocking)
            return h
            
        h_L_target = get_h(TARGET_L_NAME)
        h_R_target = get_h(TARGET_R_NAME)
        
        mat_L_start = get_matrix_from_handle(client_id, get_h(L_START))
        mat_L_p1    = get_matrix_from_handle(client_id, get_h(L_P1))
        mat_L_p2    = get_matrix_from_handle(client_id, get_h(L_P2))
        
        mat_R_start = get_matrix_from_handle(client_id, get_h(R_START))
        mat_R_p1    = get_matrix_from_handle(client_id, get_h(R_P1))
        mat_R_p2    = get_matrix_from_handle(client_id, get_h(R_P2))
        
        # ================= 阶段 1 & 2 =================
        run_independent_phase(client_id, h_L_target, h_R_target, 
                              mat_L_start, mat_L_p1, mat_R_start, mat_R_p1, 
                              DUR_PHASE_1, "阶段1")
                              
        run_independent_phase(client_id, h_L_target, h_R_target, 
                              mat_L_p1, mat_L_p2, mat_R_p1, mat_R_p2, 
                              DUR_PHASE_2, "阶段2")
                              
        # ================= 协同准备 =================
        print("计算协同偏移量...")
        pos_L_p2, _ = decompose_matrix(mat_L_p2)
        pos_R_p2, _ = decompose_matrix(mat_R_p2)
        obj_init_pos = (pos_L_p2 + pos_R_p2) / 2.0
        
        # 创建物体初始矩阵
        mat_obj_start = create_matrix(obj_init_pos, OBJ_INIT_ORI)
        
        # 计算 Offset
        mat_obj_inv = np.linalg.inv(mat_obj_start)
        offset_L = np.dot(mat_obj_inv, mat_L_p2)
        offset_R = np.dot(mat_obj_inv, mat_R_p2)
        
        mat_obj_t1 = create_matrix(OBJ_T1_POS, OBJ_T1_ORI)
        mat_obj_t2 = create_matrix(OBJ_T2_POS, OBJ_T2_ORI)
        
        # ================= 阶段 3 & 4 =================
        run_collaborative_phase(client_id, h_L_target, h_R_target,
                                mat_obj_start, mat_obj_t1, offset_L, offset_R,
                                DUR_PHASE_3, "阶段3")
                                
        run_collaborative_phase(client_id, h_L_target, h_R_target,
                                mat_obj_t1, mat_obj_t2, offset_L, offset_R,
                                DUR_PHASE_4, "阶段4")

        print("所有任务完成。")
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