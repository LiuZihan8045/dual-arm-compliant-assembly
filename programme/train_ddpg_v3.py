import sim
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# =================================================================
# 1. 全局配置与阻抗参数
# =================================================================
# M, K, B 参数 (此处可调节机械臂软硬)
IMPEDANCE_PARAMS = {
    'M': np.diag([2.0, 2.0, 2.0]),       # 虚拟质量 2kg
    'K': np.diag([500.0, 500.0, 500.0]), # 刚度 500 N/m (越小越软)
    'B': np.diag([50.0, 50.0, 50.0])     # 阻尼 50 Ns/m (越大越粘)
}

# 场景对象名称
NAME_L_P2 = "UR10_L_p2"
NAME_R_P2 = "UR10_R_p2"
NAME_PART = "Part1_Reference"
NAME_L_TARGET = "/UR10_L_target"
NAME_R_TARGET = "/UR10_R_target"
NAME_L_SENSOR = "UR10_L_forceSensor"
NAME_R_SENSOR = "UR10_R_forceSensor" # [新增] 右臂传感器
PREFIX_L_JOINT = "JL"
PREFIX_R_JOINT = "JR"

# 任务参数
PART_GOAL_POS = np.array([0.0, -0.775, 0.220])
PART_GOAL_ORI = np.array([0, 0, 0]) 
PART_INIT_ORI_EULER = np.array([0, 0, 0])

DURATION = 10.0
STEP_SIZE = 0.05
TOTAL_STEPS = int(DURATION / STEP_SIZE)

# =================================================================
# 2. 数学模型类 (用于计算关节力矩)
# =================================================================
class UR10_MathModel:
    """UR10 运动学与静力学模型"""
    def __init__(self, base_offset_xyz):
        # Modified DH 参数 [alpha, a, d, theta_offset]
        self.dh_params = [
            [0,       0,       0.1273,  0],
            [np.pi/2, 0,       0,       0],
            [0,       -0.612,  0,       0],
            [0,       -0.5723, 0,       0],
            [np.pi/2, 0,       0.1639,  0],
            [-np.pi/2,0,       0.1157,  0]
        ]
        self.base_pos = np.array(base_offset_xyz)

    def forward_kinematics(self, joints):
        T = np.eye(4)
        T[:3, 3] = self.base_pos
        for i, (alpha, a, d, offset) in enumerate(self.dh_params):
            q = joints[i] + offset
            ct, st = np.cos(q), np.sin(q)
            ca, sa = np.cos(alpha), np.sin(alpha)
            Ti = np.array([
                [ct, -st, 0, a],
                [st*ca, ct*ca, -sa, -sa*d],
                [st*sa, ct*sa, ca, ca*d],
                [0, 0, 0, 1]
            ])
            T = T @ Ti
        return T

    def get_jacobian(self, joints):
        """数值微分法计算雅可比"""
        n = 6; epsilon = 1e-4; J = np.zeros((6, n))
        T_current = self.forward_kinematics(joints)
        pos_current = T_current[:3, 3]
        rot_current = R.from_matrix(T_current[:3, :3]).as_euler('xyz')
        
        for i in range(n):
            q_p = np.array(joints); q_p[i] += epsilon
            T_new = self.forward_kinematics(q_p)
            J[:3, i] = (T_new[:3, 3] - pos_current) / epsilon
            rot_new = R.from_matrix(T_new[:3, :3]).as_euler('xyz')
            rot_diff = (rot_new - rot_current + np.pi) % (2*np.pi) - np.pi
            J[3:, i] = rot_diff / epsilon
        return J

    def compute_torques(self, joints, force_vec):
        """Tau = J^T * F"""
        J = self.get_jacobian(joints)
        # 构造 6维 Wrench (力+力矩), 假设传感器力矩为0简化计算
        wrench = np.concatenate([force_vec, [0,0,0]])
        return J.T @ wrench

# =================================================================
# 3. 核心控制类
# =================================================================
class AdmittanceController:
    def __init__(self, M, K, B, dt):
        self.M, self.K, self.B, self.dt = M, K, B, dt
        self.offset_pos = np.zeros(3)
        self.offset_vel = np.zeros(3)

    def update(self, f_ext):
        m_inv = np.linalg.inv(self.M)
        acc = m_inv @ (f_ext - self.B @ self.offset_vel - self.K @ self.offset_pos)
        self.offset_vel += acc * self.dt
        self.offset_pos += self.offset_vel * self.dt
        return self.offset_pos

class TrajectoryPlanner:
    def __init__(self, start_pos, start_quat, goal_pos, goal_euler, total_steps):
        self.total_steps = total_steps
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.slerp = Slerp([0, 1], R.from_matrix([
            R.from_quat(start_quat).as_matrix(), 
            R.from_euler('xyz', goal_euler).as_matrix()
        ]))

    def get_ideal_pose(self, step):
        t = min(step / self.total_steps, 1.0)
        pos = (1 - t) * self.start_pos + t * self.goal_pos
        mat = np.eye(4)
        mat[:3, 3] = pos
        mat[:3, :3] = self.slerp([t]).as_matrix()[0]
        return mat

# =================================================================
# 4. 辅助函数
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

def get_joint_positions(cid, handles):
    return [sim.simxGetJointPosition(cid, h, sim.simx_opmode_blocking)[1] for h in handles]

# =================================================================
# 5. 主流程
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
    h_r_sensor = sim.simxGetObjectHandle(cid, NAME_R_SENSOR, sim.simx_opmode_blocking)[1]
    
    h_j_l = [sim.simxGetObjectHandle(cid, f"{PREFIX_L_JOINT}{i}", sim.simx_opmode_blocking)[1] for i in range(1,7)]
    h_j_r = [sim.simxGetObjectHandle(cid, f"{PREFIX_R_JOINT}{i}", sim.simx_opmode_blocking)[1] for i in range(1,7)]

    # 初始化数学模型
    robot_l = UR10_MathModel([-0.525, 0, 0]) # 左臂基座坐标
    robot_r = UR10_MathModel([0.525, 0, 0])  # 右臂基座坐标

    # 初始化对齐
    print(">>> 系统初始化...")
    m_l_p2 = get_matrix_from_handle(cid, h_l_p2)
    m_r_p2 = get_matrix_from_handle(cid, h_r_p2)
    init_pos = (m_l_p2[:3, 3] + m_r_p2[:3, 3]) / 2.0
    m_start = np.eye(4); m_start[:3,3] = init_pos
    m_start[:3,:3] = R.from_euler('xyz', PART_INIT_ORI_EULER).as_matrix()
    
    set_matrix(cid, h_part, m_start)
    sim.simxSynchronousTrigger(cid)
    
    # 标定重力
    print(">>> 重力标定中...")
    bias_l_list, bias_r_list = [], []
    for _ in range(20):
        bias_l_list.append(get_force_data(cid, h_l_sensor))
        bias_r_list.append(get_force_data(cid, h_r_sensor))
        sim.simxSynchronousTrigger(cid)
    bias_l = np.mean(bias_l_list, axis=0)
    bias_r = np.mean(bias_r_list, axis=0)
    print(f"    左臂偏置: {np.linalg.norm(bias_l):.2f}N | 右臂偏置: {np.linalg.norm(bias_r):.2f}N")

    # 计算 Offset
    inv_part = np.linalg.inv(m_start)
    offset_l = np.dot(inv_part, m_l_p2)
    offset_r = np.dot(inv_part, m_r_p2)

    # 启动控制器
    start_quat = R.from_matrix(m_start[:3,:3]).as_quat()
    planner = TrajectoryPlanner(init_pos, start_quat, PART_GOAL_POS, PART_GOAL_ORI, TOTAL_STEPS)
    admittance = AdmittanceController(IMPEDANCE_PARAMS['M'], IMPEDANCE_PARAMS['K'], IMPEDANCE_PARAMS['B'], STEP_SIZE)

    print("\n" + "="*80)
    print(f"{'Step':<5}|{'Err(mm)':<8}|{'L_Force':<8}|{'R_Force':<8}|{'L_Tau(Nm) [J1,J2,J3...]':<30}|{'R_Tau(Nm)':<10}")
    print("="*80)

    try:
        for step in range(TOTAL_STEPS + 1):
            # 1. 轨迹规划
            m_ideal = planner.get_ideal_pose(step)
            
            # 2. 读力 & 干扰注入
            f_l_raw = get_force_data(cid, h_l_sensor)
            f_r_raw = get_force_data(cid, h_r_sensor) # 读取右臂力
            f_ext_l = f_l_raw - bias_l
            
            # 虚拟干扰 (Step 100-120 施加 30N)
            f_total = f_ext_l
            if 100 <= step <= 120:
                f_total += np.array([0, 30.0, 0])
            
            # 3. 导纳修正
            delta_pos = admittance.update(f_total)
            m_cmd = m_ideal.copy()
            m_cmd[:3, 3] += delta_pos
            
            # 4. 执行
            set_matrix(cid, h_part, m_cmd)
            set_matrix(cid, h_l_target, np.dot(m_cmd, offset_l))
            set_matrix(cid, h_r_target, np.dot(m_cmd, offset_r))
            sim.simxSynchronousTrigger(cid)
            
            # 5. 数据计算 (力矩 & 误差)
            pos_err = np.linalg.norm(delta_pos) * 1000
            
            # 读取关节角
            q_l = get_joint_positions(cid, h_j_l)
            q_r = get_joint_positions(cid, h_j_r)
            
            # 计算力矩 (左臂承担大部分力，右臂简单计算)
            # 注意：这里将 f_total 映射到关节空间
            tau_l = robot_l.compute_torques(q_l, f_total) 
            tau_r = robot_r.compute_torques(q_r, f_r_raw - bias_r)
            
            # 6. 实时显示 (每5步刷新一次，防止刷屏太快看不清)
            if step % 2 == 0:
                f_l_mag = np.linalg.norm(f_total)
                f_r_mag = np.linalg.norm(f_r_raw - bias_r)
                
                # 格式化力矩字符串 (只显示前3个关节)
                tau_l_str = f"[{tau_l[0]:5.1f},{tau_l[1]:5.1f},{tau_l[2]:5.1f}]"
                tau_r_str = f"[{tau_r[0]:5.1f},{tau_r[1]:5.1f},{tau_r[2]:5.1f}]"
                
                print(f"{step:<5}|{pos_err:<8.2f}|{f_l_mag:<8.1f}|{f_r_mag:<8.1f}|{tau_l_str:<30}|{tau_r_str:<10}")

        print("\n>>> 任务完成")
        for _ in range(50): sim.simxSynchronousTrigger(cid)

    except KeyboardInterrupt:
        print("\n停止")
    finally:
        sim.simxStopSimulation(cid, sim.simx_opmode_blocking)
        sim.simxFinish(cid)

if __name__ == "__main__":
    main()