import sim
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# =================================================================
# 1. 全局配置与物理参数
# =================================================================
# V-REP 对象名称
NAME_L_P2 = "UR10_L_p2"
NAME_R_P2 = "UR10_R_p2"
NAME_PART = "Part1_Reference"
NAME_L_TARGET = "/UR10_L_target"
NAME_R_TARGET = "/UR10_R_target"
PREFIX_L_JOINT = "JL"
PREFIX_R_JOINT = "JR"

# 任务参数
PART_GOAL_POS = np.array([0.0, -0.775, 0.220])
PART_GOAL_ORI = np.array([0, 0, 0])
PART_INIT_ORI_EULER = np.array([0, 0, 0])

# 仿真参数
DURATION = 10.0
STEP_SIZE = 0.05
TOTAL_STEPS = int(DURATION / STEP_SIZE)

# --- 动力学参数 (Python 纯数值计算用) ---
OBJ_MASS = 2.0  # 物体质量 kg
GRAVITY = np.array([0, 0, -9.81])

# 阻抗参数 (决定系统的"软硬")
M_VIRT = 5.0    # 虚拟质量
K_VIRT = 800.0  # 虚拟刚度 (N/m)
B_VIRT = 100.0  # 虚拟阻尼 (Ns/m)

# =================================================================
# 2. 数学模型类 (用于计算关节力矩)
# =================================================================
class UR10_MathModel:
    """
    UR10 运动学与静力学模型
    用于将 Python 计算出的虚拟力映射为关节力矩
    """
    def __init__(self, base_offset_x):
        # UR10 标准 DH 参数 [alpha, a, d, theta_offset]
        self.dh = [
            [0, 0, 0.1273, 0], [np.pi/2, 0, 0, 0], [0, -0.612, 0, 0],
            [0, -0.5723, 0, 0], [np.pi/2, 0, 0.1639, 0], [-np.pi/2, 0, 0.1157, 0]
        ]
        self.base_pos = np.array([base_offset_x, 0, 0])

    def forward_kine(self, q):
        T = np.eye(4); T[:3,3] = self.base_pos
        for i, (alp, a, d, off) in enumerate(self.dh):
            theta = q[i] + off
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alp), np.sin(alp)
            Ti = np.array([[ct, -st, 0, a], [st*ca, ct*ca, -sa, -sa*d], [st*sa, ct*sa, ca, ca*d], [0,0,0,1]])
            T = T @ Ti
        return T

    def get_jacobian(self, q):
        """数值微分法计算雅可比矩阵"""
        n = 6; eps = 1e-4; J = np.zeros((6, n))
        T0 = self.forward_kine(q)
        p0 = T0[:3,3]
        # 简化处理：仅计算位置雅可比用于力的映射，忽略力矩映射的复杂旋转
        for i in range(n):
            q_ = np.array(q); q_[i] += eps
            T1 = self.forward_kine(q_)
            J[:3,i] = (T1[:3,3] - p0)/eps
            # 姿态部分简化，Demo主要展示受力
        return J

    def compute_torques(self, joints, force_vec):
        """Tau = J^T * F"""
        J = self.get_jacobian(joints)
        # 构造 6维 Wrench (力+力矩0)
        wrench = np.concatenate([force_vec, [0,0,0]])
        return J.T @ wrench

# =================================================================
# 3. 虚拟动力学引擎 (核心：Python 算位置)
# =================================================================
class VirtualPhysicsEngine:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        # 恒定外力：重力
        self.g_force = OBJ_MASS * GRAVITY 

    def reset(self, p):
        self.pos = np.array(p)
        self.vel = np.zeros(3)

    def step(self, target_pos, f_disturb=np.zeros(3)):
        """
        输入: 理想轨迹点(target_pos), 扰动(f_disturb)
        输出: 实际物理位置, 产生的内力(用于算力矩)
        """
        # 1. 导纳力 (Spring-Damper): 机械臂试图把物体拉回轨迹的力
        err = target_pos - self.pos
        f_compliance = K_VIRT * err - B_VIRT * self.vel
        
        # 2. 合力 = 导纳力 + 重力 + 扰动
        # 这是作用在物体上的净力，决定物体的加速度
        f_net = f_compliance + self.g_force + f_disturb
        
        # 3. 动力学积分 (F=ma)
        acc = f_net / M_VIRT 
        self.vel += acc * STEP_SIZE
        self.pos += self.vel * STEP_SIZE
        
        # 4. 返回值
        # 机械臂受到的反作用力 = -(导纳力)
        # 这是机械臂为了维持当前位置所施加的力
        load_on_arm = -f_compliance 
        return self.pos, load_on_arm

# =================================================================
# 4. 辅助函数
# =================================================================
def connect_vrep():
    print("连接 V-REP...")
    sim.simxFinish(-1)
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if client_id == -1:
        print("连接失败")
        sys.exit()
    return client_id

def create_matrix(pos, quat):
    mat = np.eye(4); mat[:3, 3] = pos; mat[:3, :3] = R.from_quat(quat).as_matrix()
    return mat

def get_matrix_from_handle(cid, handle):
    _, pos = sim.simxGetObjectPosition(cid, handle, -1, sim.simx_opmode_blocking)
    _, quat = sim.simxGetObjectQuaternion(cid, handle, -1, sim.simx_opmode_blocking)
    return create_matrix(pos, quat)

def set_target_pose(cid, handle, mat):
    pos = mat[:3, 3]
    quat = R.from_matrix(mat[:3, :3]).as_quat()
    sim.simxSetObjectPosition(cid, handle, -1, pos.tolist(), sim.simx_opmode_oneshot)
    sim.simxSetObjectQuaternion(cid, handle, -1, quat.tolist(), sim.simx_opmode_oneshot)

def interpolate_matrix(start_mat, end_mat, t):
    pos_s = start_mat[:3, 3]; pos_e = end_mat[:3, 3]
    curr_pos = (1 - t) * pos_s + t * pos_e
    rot_s = R.from_matrix(start_mat[:3, :3]); rot_e = R.from_matrix(end_mat[:3, :3])
    slerp = Slerp([0, 1], R.from_matrix([rot_s.as_matrix(), rot_e.as_matrix()]))
    curr_rot = slerp([t]).as_matrix()[0]
    mat = np.eye(4); mat[:3, 3] = curr_pos; mat[:3, :3] = curr_rot
    return mat

def get_joint_angles(cid, handles):
    return [sim.simxGetJointPosition(cid, h, sim.simx_opmode_blocking)[1] for h in handles]

def print_dashboard(step, err, load_vec, tau_l, tau_r, disturbed):
    bar_len = 20
    load_mag = np.linalg.norm(load_vec)
    bar = '█' * int(min(load_mag/2.0, bar_len)) + '-' * int(max(bar_len - load_mag/2.0, 0))
    st = "[!!! 扰动 !!!]" if disturbed else "             "
    
    # 简单的格式化输出
    print(f"Step {step:03d} {st} | 误差: {err*1000:5.1f}mm | 合力: {load_mag:5.1f}N |{bar}|")
    print(f"   L_Tau (Nm): [{tau_l[0]:5.1f}, {tau_l[1]:5.1f}, {tau_l[2]:5.1f}]")
    print(f"   R_Tau (Nm): [{tau_r[0]:5.1f}, {tau_r[1]:5.1f}, {tau_r[2]:5.1f}]")
    print("-" * 60)

# =================================================================
# 5. 主流程
# =================================================================
def main():
    cid = connect_vrep()
    sim.simxSynchronous(cid, True)
    sim.simxStartSimulation(cid, sim.simx_opmode_blocking)
    
    try:
        # --- 1. 获取句柄 ---
        h_part = sim.simxGetObjectHandle(cid, NAME_PART, sim.simx_opmode_blocking)[1]
        h_l_target = sim.simxGetObjectHandle(cid, NAME_L_TARGET, sim.simx_opmode_blocking)[1]
        h_r_target = sim.simxGetObjectHandle(cid, NAME_R_TARGET, sim.simx_opmode_blocking)[1]
        h_l_p2 = sim.simxGetObjectHandle(cid, NAME_L_P2, sim.simx_opmode_blocking)[1]
        h_r_p2 = sim.simxGetObjectHandle(cid, NAME_R_P2, sim.simx_opmode_blocking)[1]
        
        # 关节句柄 (用于读取角度算力矩)
        hj_l = [sim.simxGetObjectHandle(cid, f"{PREFIX_L_JOINT}{i}", sim.simx_opmode_blocking)[1] for i in range(1,7)]
        hj_r = [sim.simxGetObjectHandle(cid, f"{PREFIX_R_JOINT}{i}", sim.simx_opmode_blocking)[1] for i in range(1,7)]

        # --- 2. 初始化对齐与数学模型 ---
        print(">>> 初始化系统...")
        m_l_p2 = get_matrix_from_handle(cid, h_l_p2)
        m_r_p2 = get_matrix_from_handle(cid, h_r_p2)
        
        # 初始中点
        init_pos = (m_l_p2[:3, 3] + m_r_p2[:3, 3]) / 2.0
        m_start = np.eye(4); m_start[:3, 3] = init_pos
        m_start[:3, :3] = R.from_euler('xyz', PART_INIT_ORI_EULER).as_matrix()
        
        # V-REP 对齐
        set_target_pose(cid, h_part, m_start)
        sim.simxSynchronousTrigger(cid)
        
        # Offset 锁定
        inv_part = np.linalg.inv(m_start)
        off_l = np.dot(inv_part, m_l_p2)
        off_r = np.dot(inv_part, m_r_p2)

        # 动力学模型初始化
        robot_l = UR10_MathModel(-0.525)
        robot_r = UR10_MathModel(0.525)
        physics = VirtualPhysicsEngine()
        physics.reset(init_pos)

        # 目标设置
        m_goal = create_matrix(PART_GOAL_POS, R.from_euler('xyz', PART_GOAL_ORI).as_quat())

        print(">>> 开始动力学仿真 (Python 计算 -> V-REP 显示)")
        
        # --- 3. 仿真循环 ---
        disturb_step = 100 # 第100步施加扰动

        for step in range(TOTAL_STEPS + 1):
            t = step / TOTAL_STEPS
            
            # A. 理想轨迹规划 (Slerp)
            # m_ideal 是不考虑任何力学因素的纯几何位置
            m_ideal = interpolate_matrix(m_start, m_goal, t)
            ideal_pos = m_ideal[:3, 3]
            
            # B. 扰动注入 (模拟外部冲击)
            f_disturb = np.zeros(3)
            is_disturbed = False
            if step == disturb_step:
                f_disturb = np.array([0, 50.0, 0]) # Y轴 50N 冲击
                physics.pos += np.array([0.05, 0, 0]) # 或者直接位置突变
                is_disturbed = True
            
            # C. Python 动力学解算 (核心)
            # 根据理想位置、当前物理状态、重力、扰动，算出物体的"真实"位置
            real_pos, virtual_load = physics.step(ideal_pos, f_disturb)
            
            # D. 驱动 V-REP (仅显示)
            # 将算出来的"真实"位置发给 V-REP，让它画出来
            m_real = m_ideal.copy()
            m_real[:3, 3] = real_pos # 覆盖位置，保留姿态
            
            set_target_pose(cid, h_part, m_real)
            # 协同更新双臂 Target
            set_target_pose(cid, h_l_target, np.dot(m_real, off_l))
            set_target_pose(cid, h_r_target, np.dot(m_real, off_r))
            
            sim.simxSynchronousTrigger(cid)
            
            # E. 数据计算与论证
            # 1. 位置误差
            pos_err = np.linalg.norm(ideal_pos - real_pos)
            
            # 2. 关节力矩计算
            # 获取当前 V-REP 里的关节角度 (仅用于计算雅可比，不参与控制)
            q_l = get_joint_angles(cid, hj_l)
            q_r = get_joint_angles(cid, hj_r)
            
            # 假设负载平分给双臂
            half_load = virtual_load / 2.0
            
            tau_l = robot_l.compute_torques(q_l, half_load)
            tau_r = robot_r.compute_torques(q_r, half_load)
            
            # F. 输出数据
            if step % 5 == 0 or is_disturbed:
                print_dashboard(step, pos_err, virtual_load, tau_l, tau_r, is_disturbed)
            
            time.sleep(0.02)
            
        print("搬运完成")
        for _ in range(50): sim.simxSynchronousTrigger(cid)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sim.simxStopSimulation(cid, sim.simx_opmode_blocking)
        sim.simxFinish(cid)

if __name__ == "__main__":
    main()