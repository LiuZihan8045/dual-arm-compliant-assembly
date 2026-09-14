import sim
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# ================= 配置区域 =================
NAME_L_FORCE = "UR10_L_forceSensor"
NAME_R_FORCE = "UR10_R_forceSensor"
NAME_PART_REF = "Part1_Reference"
NAME_L_TARGET = "/UR10_L_target"
NAME_R_TARGET = "/UR10_R_target"
NAME_L_P2 = "UR10_L_p2"
NAME_R_P2 = "UR10_R_p2"

# 目标位姿
PART_GOAL_POS = [0.0, -0.775, 0.220] 
PART_GOAL_ORI = [0, 0, 0] 
PART_INIT_ORI = [0, 0, 0]

# 仿真参数
STEP_SIZE = 0.05
DURATION = 20.0

# ================= 导纳控制器类 =================
class AdmittanceController:
    """
    位置型阻抗控制器 (Admittance Controller)
    RL 未来将通过修改 self.K 和 self.B 来改变机械臂的“手感”
    """
    def __init__(self, dt=0.05):
        self.dt = dt
        # 阻抗参数 (初始值)
        self.M = np.diag([1.0, 1.0, 1.0])       # 虚拟质量
        self.K = np.diag([500.0, 500.0, 500.0]) # 虚拟刚度 (RL调优重点)
        self.B = np.diag([50.0, 50.0, 50.0])    # 虚拟阻尼 (RL调优重点)
        
        # 状态变量 (位移偏差及其导数)
        self.delta_x = np.zeros(3)
        self.delta_x_dot = np.zeros(3)

    def update(self, force_ext, k_val=None, b_val=None):
        """
        k_val, b_val 为 RL 输出的修正系数 (可选)
        """
        K = self.K if k_val is None else self.K * k_val
        B = self.B if b_val is None else self.B * b_val
        
        # 求解加速度: M*ddx + B*dx + K*x = F
        # ddx = M^-1 * (F - B*dx - K*x)
        acc = np.linalg.inv(self.M) @ (force_ext - B @ self.delta_x_dot - K @ self.delta_x)
        
        # 积分得到速度和位移
        self.delta_x_dot += acc * self.dt
        self.delta_x += self.delta_x_dot * self.dt
        
        return self.delta_x

# ================= 辅助函数 =================
def get_matrix(cid, handle):
    _, p = sim.simxGetObjectPosition(cid, handle, -1, sim.simx_opmode_blocking)
    _, q = sim.simxGetObjectQuaternion(cid, handle, -1, sim.simx_opmode_blocking)
    m = np.eye(4)
    m[:3, 3] = p
    m[:3, :3] = R.from_quat(q).as_matrix()
    return m

def set_matrix(cid, handle, mat):
    p = mat[:3, 3]
    q = R.from_matrix(mat[:3, :3]).as_quat()
    sim.simxSetObjectPosition(cid, handle, -1, p.tolist(), sim.simx_opmode_oneshot)
    sim.simxSetObjectQuaternion(cid, handle, -1, q.tolist(), sim.simx_opmode_oneshot)

def read_force(cid, h_sensor):
    # 读取六维力传感器
    res, state, f, t = sim.simxReadForceSensor(cid, h_sensor, sim.simx_opmode_blocking)
    if res == 0 and (state & 1):
        return np.array(f)
    return np.zeros(3)

def interpolate_pose(start_m, end_m, t):
    # 位置线性插值
    pos = (1-t)*start_m[:3,3] + t*end_m[:3,3]
    # 姿态 SLERP
    rots = R.from_matrix([start_m[:3,:3], end_m[:3,:3]])
    slerp = Slerp([0, 1], rots)
    rot = slerp([t]).as_matrix()[0]
    m = np.eye(4)
    m[:3,3], m[:3,:3] = pos, rot
    return m

# ================= 主程序 =================
def main():
    sim.simxFinish(-1)
    cid = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if cid == -1: return print("连接失败")
    sim.simxSynchronous(cid, True)
    sim.simxStartSimulation(cid, sim.simx_opmode_blocking)

    # 获取句柄
    h_part = sim.simxGetObjectHandle(cid, NAME_PART_REF, sim.simx_opmode_blocking)[1]
    h_l_target = sim.simxGetObjectHandle(cid, NAME_L_TARGET, sim.simx_opmode_blocking)[1]
    h_r_target = sim.simxGetObjectHandle(cid, NAME_R_TARGET, sim.simx_opmode_blocking)[1]
    h_l_force = sim.simxGetObjectHandle(cid, NAME_L_FORCE, sim.simx_opmode_blocking)[1]
    h_r_force = sim.simxGetObjectHandle(cid, NAME_R_FORCE, sim.simx_opmode_blocking)[1]
    h_l_p2 = sim.simxGetObjectHandle(cid, NAME_L_P2, sim.simx_opmode_blocking)[1]
    h_r_p2 = sim.simxGetObjectHandle(cid, NAME_R_P2, sim.simx_opmode_blocking)[1]

    # 初始化 Admittance 控制器
    adm_l = AdmittanceController(dt=STEP_SIZE)
    adm_r = AdmittanceController(dt=STEP_SIZE)

    # 初始对齐与 Offset 计算
    m_l_p2 = get_matrix(cid, h_l_p2)
    m_r_p2 = get_matrix(cid, h_r_p2)
    init_pos = (m_l_p2[:3, 3] + m_r_p2[:3, 3]) / 2.0
    m_start = np.eye(4)
    m_start[:3,3] = init_pos
    m_start[:3,:3] = R.from_euler('xyz', PART_INIT_ORI).as_matrix()
    
    set_matrix(cid, h_part, m_start)
    sim.simxSynchronousTrigger(cid)
    
    inv_start = np.linalg.inv(m_start)
    offset_l = inv_start @ m_l_p2
    offset_r = inv_start @ m_r_p2

    # 目标矩阵
    m_goal = np.eye(4)
    m_goal[:3,3] = PART_GOAL_POS
    m_goal[:3,:3] = R.from_euler('xyz', PART_GOAL_ORI).as_matrix()

    steps = int(DURATION / STEP_SIZE)
    print("开始变阻抗协同搬运...")

    for i in range(steps + 1):
        t = i / steps
        # 1. 获取 SLERP 理想位姿
        m_desired = interpolate_pose(m_start, m_goal, t)
        
        # 2. 读取传感器力 (虽然你现在没设碰撞，但逻辑要打通)
        f_l = read_force(cid, h_l_force)
        f_r = read_force(cid, h_r_force)
        
        # 3. 计算阻抗位移修正 (未来 RL 会在这里输出 k_gain, b_gain)
        dx_l = adm_l.update(f_l)
        dx_r = adm_r.update(f_r)
        
        # 4. 物体中心跟随理想位姿 (这里假设物体中心不随阻抗漂移，或者两臂协同修正)
        # 通常 Admittance 作用在末端执行器上
        set_matrix(cid, h_part, m_desired)
        
        # 5. 反算 Target 并在其基础上增加阻抗偏差
        m_l_cmd = m_desired @ offset_l
        m_r_cmd = m_desired @ offset_r
        
        m_l_cmd[:3, 3] += dx_l # 加上左臂的阻抗退让
        m_r_cmd[:3, 3] += dx_r # 加上右臂的阻抗退让
        
        set_matrix(cid, h_l_target, m_l_cmd)
        set_matrix(cid, h_r_target, m_r_cmd)
        
        sim.simxSynchronousTrigger(cid)

    print("搬运结束")
    sim.simxStopSimulation(cid, sim.simx_opmode_blocking)
    sim.simxFinish(cid)

if __name__ == "__main__":
    main()