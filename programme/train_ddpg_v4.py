import sim
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# =================================================================
# 1. UR10 机器人数学模型 (保持不变)
# =================================================================
class UR10_MathModel:
    def __init__(self, base_offset_x):
        self.dh_params = [
            [0,       0,       0.1273,  0],
            [np.pi/2, 0,       0,       0],
            [0,       -0.612,  0,       0],
            [0,       -0.5723, 0,       0],
            [np.pi/2, 0,       0.1639,  0],
            [-np.pi/2,0,       0.1157,  0]
        ]
        self.base_pos = np.array([base_offset_x, 0, 0])

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
        n = 6; epsilon = 1e-4; J = np.zeros((6, n))
        T_current = self.forward_kinematics(joints)
        pos_current = T_current[:3, 3]
        rot_current = R.from_matrix(T_current[:3, :3]).as_euler('xyz')
        for i in range(n):
            q_perturbed = np.array(joints); q_perturbed[i] += epsilon
            T_new = self.forward_kinematics(q_perturbed)
            J[:3, i] = (T_new[:3, 3] - pos_current) / epsilon
            rot_new = R.from_matrix(T_new[:3, :3]).as_euler('xyz')
            rot_diff = (rot_new - rot_current + np.pi) % (2 * np.pi) - np.pi
            J[3:, i] = rot_diff / epsilon
        return J

    def compute_torques(self, joints, force_wrench):
        J = self.get_jacobian(joints)
        return J.T @ force_wrench

# =================================================================
# 2. 虚拟动力学引擎 (保持不变)
# =================================================================
class VirtualDynamicsEngine:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.M = 5.0; self.B = 100.0; self.K = 800.0
        self.pos = np.zeros(3); self.vel = np.zeros(3)
        self.internal_force = np.zeros(3)

    def reset(self, init_pos):
        self.pos = np.array(init_pos); self.vel = np.zeros(3)

    def step(self, target_pos, disturbance_force=np.zeros(3)):
        error = target_pos - self.pos
        f_spring = self.K * error
        f_damper = -self.B * self.vel
        total_force = f_spring + f_damper + disturbance_force
        self.internal_force = total_force
        acc = total_force / self.M
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt
        return self.pos, total_force

# =================================================================
# 3. 辅助函数 (新增矩阵操作)
# =================================================================
def get_joint_angles(cid, handles):
    return [sim.simxGetJointPosition(cid, h, sim.simx_opmode_blocking)[1] for h in handles]

# [新增] 获取矩阵工具
def get_matrix_from_pose(pos, quat):
    m = np.eye(4)
    m[:3, 3] = pos
    m[:3, :3] = R.from_quat(quat).as_matrix()
    return m

# [新增] 获取VREP对象矩阵
def get_matrix(cid, handle):
    _, p = sim.simxGetObjectPosition(cid, handle, -1, sim.simx_opmode_blocking)
    _, q = sim.simxGetObjectQuaternion(cid, handle, -1, sim.simx_opmode_blocking)
    return get_matrix_from_pose(p, q)

# [新增] 设置VREP对象矩阵
def set_matrix(cid, handle, mat):
    p = mat[:3, 3]
    q = R.from_matrix(mat[:3, :3]).as_quat()
    sim.simxSetObjectPosition(cid, handle, -1, p.tolist(), sim.simx_opmode_oneshot)
    sim.simxSetObjectQuaternion(cid, handle, -1, q.tolist(), sim.simx_opmode_oneshot)

def print_dashboard(step, err, force, torques_l, torques_r, disturbed):
    bar_len = 20
    force_mag = np.linalg.norm(force)
    bar = '█' * int(min(force_mag/5.0, bar_len)) + '-' * int(max(bar_len - force_mag/5.0, 0))
    status = " [!!! 扰动生效 !!!]" if disturbed else ""
    
    print(f"\n{'-'*30} Step {step:02d} {status} {'-'*30}")
    print(f"1. 轨迹跟踪 | 位置误差: {err*1000:.2f} mm | 双臂合力: {force_mag:.2f} N |{bar}|")
    print(f"2. 关节力矩 (Nm) [前3关节]:")
    print(f"   Left : [{torques_l[0]:6.1f}, {torques_l[1]:6.1f}, {torques_l[2]:6.1f}]")
    print(f"   Right: [{torques_r[0]:6.1f}, {torques_r[1]:6.1f}, {torques_r[2]:6.1f}]")
    sys.stdout.flush()

# =================================================================
# 4. 主流程 (修复 Target 设置逻辑)
# =================================================================
def main():
    sim.simxFinish(-1)
    cid = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if cid == -1: return print("V-REP 连接失败")
    sim.simxSynchronous(cid, True)
    sim.simxStartSimulation(cid, sim.simx_opmode_blocking)

    # 句柄
    h_l_target = sim.simxGetObjectHandle(cid, "/UR10_L_target", sim.simx_opmode_blocking)[1]
    h_r_target = sim.simxGetObjectHandle(cid, "/UR10_R_target", sim.simx_opmode_blocking)[1]
    h_part_ref = sim.simxGetObjectHandle(cid, "Part1_Reference", sim.simx_opmode_blocking)[1]
    h_l_p2 = sim.simxGetObjectHandle(cid, "UR10_L_p2", sim.simx_opmode_blocking)[1]
    h_r_p2 = sim.simxGetObjectHandle(cid, "UR10_R_p2", sim.simx_opmode_blocking)[1]
    
    h_joints_l = [sim.simxGetObjectHandle(cid, f"JL{i}", sim.simx_opmode_blocking)[1] for i in range(1,7)]
    h_joints_r = [sim.simxGetObjectHandle(cid, f"JR{i}", sim.simx_opmode_blocking)[1] for i in range(1,7)]

    # 数学模型
    robot_l = UR10_MathModel(base_offset_x=-0.525)
    robot_r = UR10_MathModel(base_offset_x=0.525)
    dynamics = VirtualDynamicsEngine(dt=0.05)

    # --- [关键修复] 初始化与 Offset 计算 ---
    print(">>> 正在计算双臂抓取 Offset...")
    m_l_p2 = get_matrix(cid, h_l_p2)
    m_r_p2 = get_matrix(cid, h_r_p2)
    
    # 1. 算出物体应该在的初始位置 (中点)
    init_pos = (m_l_p2[:3, 3] + m_r_p2[:3, 3]) / 2.0
    
    # 2. 构建物体初始矩阵 (位置居中，姿态归零)
    m_obj_start = np.eye(4)
    m_obj_start[:3, 3] = init_pos
    m_obj_start[:3, :3] = R.from_euler('xyz', [0,0,0]).as_matrix() # 假设初始姿态平正
    
    # 3. 将物体移到初始点
    set_matrix(cid, h_part_ref, m_obj_start)
    sim.simxSynchronousTrigger(cid)
    
    # 4. 计算左右手相对于物体的固定变换矩阵 (Offset)
    # Offset = Inv(Object) * Hand_P2
    m_obj_inv = np.linalg.inv(m_obj_start)
    offset_l = m_obj_inv @ m_l_p2
    offset_r = m_obj_inv @ m_r_p2

    # --- 轨迹规划设置 ---
    # 为了演示简单，只做位置变化，姿态保持初始
    start_pos = init_pos
    end_pos   = start_pos + np.array([0.0, 0.0, -0.3]) # 向下压 30cm
    dynamics.reset(start_pos)

    total_steps = 40
    disturbance_step = 20
    
    try:
        for step in range(total_steps):
            # 1. 理想轨迹点
            t = step / total_steps
            ideal_pos = (1-t)*start_pos + t*end_pos
            
            # 2. 扰动
            f_disturb = np.zeros(3)
            is_disturbed = False
            if step == disturbance_step:
                dynamics.pos += np.array([0.05, 0.0, 0.0]) # 突发 50mm 偏移
                is_disturbed = True
            
            # 3. 动力学解算 (得到物体中心位置)
            real_pos, virtual_force = dynamics.step(ideal_pos, f_disturb)
            
            # --- [关键修复] 构建物体矩阵并反算两臂 Target ---
            m_obj_current = np.eye(4)
            m_obj_current[:3, 3] = real_pos
            m_obj_current[:3, :3] = m_obj_start[:3, :3] # 姿态保持不变
            
            # 反算两臂各自的目标位置
            m_target_l = m_obj_current @ offset_l
            m_target_r = m_obj_current @ offset_r
            
            # 4. 更新 V-REP
            set_matrix(cid, h_part_ref, m_obj_current)
            set_matrix(cid, h_l_target, m_target_l)
            set_matrix(cid, h_r_target, m_target_r)
            
            sim.simxSynchronousTrigger(cid)
            
            # 5. 数据计算
            q_l = get_joint_angles(cid, h_joints_l)
            q_r = get_joint_angles(cid, h_joints_r)
            
            wrench_half = np.concatenate([virtual_force / 2.0, [0,0,0]])
            tau_l = robot_l.compute_torques(q_l, wrench_half)
            tau_r = robot_r.compute_torques(q_r, wrench_half)
            
            pos_error = np.linalg.norm(ideal_pos - real_pos)
            print_dashboard(step, pos_error, virtual_force, tau_l, tau_r, is_disturbed)
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        sim.simxStopSimulation(cid, sim.simx_opmode_blocking)
        sim.simxFinish(cid)

if __name__ == "__main__":
    main()