import sim
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# =================================================================
# 1. 阻抗参数与全局配置
# =================================================================
# M, K, B 参数现在真的起作用了！
IMPEDANCE_PARAMS = {
    'M': np.diag([2.0, 2.0, 2.0]),      # 虚拟质量 2kg
    'K': np.diag([500.0, 500.0, 500.0]), # 刚度 500 N/m
    'B': np.diag([50.0, 50.0, 50.0])    # 阻尼 50 Ns/m
}

NAME_L_P2 = "UR10_L_p2"
NAME_R_P2 = "UR10_R_p2"
NAME_PART = "Part1_Reference"
NAME_L_TARGET = "/UR10_L_target"
NAME_R_TARGET = "/UR10_R_target"
NAME_L_SENSOR = "UR10_L_forceSensor" # [新增] 用于读取真实力
PREFIX_L_JOINT = "JL"
PREFIX_R_JOINT = "JR"

PART_GOAL_POS = np.array([0.0, -0.775, 0.220])
PART_GOAL_ORI = np.array([0, 0, 0]) 
PART_INIT_ORI_EULER = np.array([0, 0, 0])

DURATION = 10.0
STEP_SIZE = 0.05
TOTAL_STEPS = int(DURATION / STEP_SIZE)

# =================================================================
# 2. 核心类定义
# =================================================================
class AdmittanceController:
    """
    位置型阻抗控制器 (Admittance Control)
    方程: M*ddx + B*dx + K*x = F_ext
    """
    def __init__(self, M, K, B, dt):
        self.M = M
        self.K = K
        self.B = B
        self.dt = dt
        
        # 内部状态：位移偏差(x) 和 速度偏差(dx)
        # 注意：这里的 x 是指相对于理想轨迹的"偏差量"，不是绝对坐标
        self.offset_pos = np.zeros(3)
        self.offset_vel = np.zeros(3)

    def update(self, f_ext):
        """
        输入: 外部受力 F_ext (3,)
        输出: 位置修正量 delta_x (3,)
        """
        # 计算加速度: acc = M_inv * (F - B*vel - K*pos)
        m_inv = np.linalg.inv(self.M)
        damping = self.B @ self.offset_vel
        stiffness = self.K @ self.offset_pos
        
        acc = m_inv @ (f_ext - damping - stiffness)
        
        # 欧拉积分
        self.offset_vel += acc * self.dt
        self.offset_pos += self.offset_vel * self.dt
        
        return self.offset_pos

class TrajectoryPlanner:
    def __init__(self, start_pos, start_quat, goal_pos, goal_euler, total_steps):
        self.total_steps = total_steps
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        r_start = R.from_quat(start_quat)
        r_goal = R.from_euler('xyz', goal_euler)
        self.slerp = Slerp([0, 1], R.from_matrix([r_start.as_matrix(), r_goal.as_matrix()]))

    def get_ideal_pose(self, step):
        t = step / self.total_steps
        if t > 1.0: t = 1.0
        pos = (1 - t) * self.start_pos + t * self.goal_pos
        rot_mat = self.slerp([t]).as_matrix()[0]
        mat = np.eye(4)
        mat[:3, 3] = pos
        mat[:3, :3] = rot_mat
        return mat

# =================================================================
# 3. 辅助函数
# =================================================================
def connect_vrep():
    sim.simxFinish(-1)
    cid = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if cid == -1: sys.exit("连接失败")
    sim.simxSynchronous(cid, True)
    sim.simxStartSimulation(cid, sim.simx_opmode_blocking)
    return cid

def get_matrix_from_handle(cid, handle):
    _, p = sim.simxGetObjectPosition(cid, handle, -1, sim.simx_opmode_blocking)
    _, q = sim.simxGetObjectQuaternion(cid, handle, -1, sim.simx_opmode_blocking)
    m = np.eye(4); m[:3,3] = p; m[:3,:3] = R.from_quat(q).as_matrix()
    return m

def set_matrix(cid, handle, mat):
    p = mat[:3,3]; q = R.from_matrix(mat[:3,:3]).as_quat()
    sim.simxSetObjectPosition(cid, handle, -1, p.tolist(), sim.simx_opmode_oneshot)
    sim.simxSetObjectQuaternion(cid, handle, -1, q.tolist(), sim.simx_opmode_oneshot)

def get_force_data(cid, handle):
    res, state, f, _ = sim.simxReadForceSensor(cid, handle, sim.simx_opmode_blocking)
    if res == 0 and (state & 1): return np.array(f)
    return np.zeros(3)

# =================================================================
# 4. 主流程
# =================================================================
def main():
    cid = connect_vrep()
    
    # 获取句柄
    h_part = sim.simxGetObjectHandle(cid, NAME_PART, sim.simx_opmode_blocking)[1]
    h_l_target = sim.simxGetObjectHandle(cid, NAME_L_TARGET, sim.simx_opmode_blocking)[1]
    h_r_target = sim.simxGetObjectHandle(cid, NAME_R_TARGET, sim.simx_opmode_blocking)[1]
    h_l_p2 = sim.simxGetObjectHandle(cid, NAME_L_P2, sim.simx_opmode_blocking)[1]
    h_r_p2 = sim.simxGetObjectHandle(cid, NAME_R_P2, sim.simx_opmode_blocking)[1]
    h_l_sensor = sim.simxGetObjectHandle(cid, NAME_L_SENSOR, sim.simx_opmode_blocking)[1]

    # 初始化对齐
    print(">>> 初始化...")
    m_l_p2 = get_matrix_from_handle(cid, h_l_p2)
    m_r_p2 = get_matrix_from_handle(cid, h_r_p2)
    init_pos = (m_l_p2[:3, 3] + m_r_p2[:3, 3]) / 2.0
    
    m_start = np.eye(4); m_start[:3,3] = init_pos
    m_start[:3,:3] = R.from_euler('xyz', PART_INIT_ORI_EULER).as_matrix()
    set_matrix(cid, h_part, m_start)
    sim.simxSynchronousTrigger(cid)
    
    # 标定重力 (前20步取平均)
    print(">>> 正在标定重力...")
    bias_list = []
    for _ in range(20):
        f = get_force_data(cid, h_l_sensor)
        bias_list.append(f)
        sim.simxSynchronousTrigger(cid)
    bias_force = np.mean(bias_list, axis=0)
    print(f"    重力偏置: {np.round(bias_force, 2)} N")

    # 计算 Offset
    inv_part = np.linalg.inv(m_start)
    offset_l = np.dot(inv_part, m_l_p2) 
    offset_r = np.dot(inv_part, m_r_p2)

    # 轨迹与控制器初始化
    start_quat = R.from_matrix(m_start[:3,:3]).as_quat()
    planner = TrajectoryPlanner(init_pos, start_quat, PART_GOAL_POS, PART_GOAL_ORI, TOTAL_STEPS)
    
    admittance = AdmittanceController(
        IMPEDANCE_PARAMS['M'], 
        IMPEDANCE_PARAMS['K'], 
        IMPEDANCE_PARAMS['B'], 
        STEP_SIZE
    )

    print(f"\n>>> 开始变阻抗轨迹跟踪 (包含虚拟干扰测试)")
    print(f"{'Step':<5} | {'Ideal Y':<10} | {'Actual Y':<10} | {'Error(mm)':<10} | {'Force(N)':<10}")
    print("-" * 65)

    try:
        for step in range(TOTAL_STEPS + 1):
            # 1. 获取理想轨迹
            m_ideal = planner.get_ideal_pose(step)
            
            # 2. 获取真实力并计算净外力
            f_raw = get_force_data(cid, h_l_sensor)
            f_ext = f_raw - bias_force
            
            # [关键测试] 在第 100-150 步期间，人为施加 20N 的 Y 轴干扰力
            # 这模拟了机械臂被人推了一把，或者撞到了虚拟墙
            virtual_force = f_ext
            if 100 <= step <= 150:
                virtual_force += np.array([0, 20.0, 0]) 
            
            # 3. 导纳控制核心：计算位置偏差
            # 即使 f_ext 为 0，如果上面的 virtual_force 不为 0，也会产生位移
            delta_pos = admittance.update(virtual_force)
            
            # 4. 应用偏差到物体位置
            m_cmd = m_ideal.copy()
            m_cmd[:3, 3] += delta_pos # 叠加导纳修正量
            
            set_matrix(cid, h_part, m_cmd)
            set_matrix(cid, h_l_target, np.dot(m_cmd, offset_l))
            set_matrix(cid, h_r_target, np.dot(m_cmd, offset_r))
            
            sim.simxSynchronousTrigger(cid)
            
            # 5. 误差分析
            # 这里的误差 = 理想规划位置 - 实际指令位置
            # 这个误差正是导纳控制器产生的"柔顺退让"
            pos_err = np.linalg.norm(delta_pos)
            
            if step % 5 == 0:
                f_mag = np.linalg.norm(virtual_force)
                print(f"{step:<5} | {m_ideal[1,3]:<10.4f} | {m_cmd[1,3]:<10.4f} | {pos_err*1000:<10.2f} | {f_mag:<10.1f}")

        print("\n>>> 仿真结束。如果看到 Error 在中间变大然后消失，说明阻抗控制成功。")
        for _ in range(50): sim.simxSynchronousTrigger(cid)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        sim.simxStopSimulation(cid, sim.simx_opmode_blocking)
        sim.simxFinish(cid)

if __name__ == "__main__":
    main()