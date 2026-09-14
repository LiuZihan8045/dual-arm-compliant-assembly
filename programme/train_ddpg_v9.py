import sim  # 需要 V-REP/CoppeliaSim 的 legacy remote API 文件 (sim.py, simConst.py, remoteApi.dll/so)
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt

# =================================================================
# 1. 全局配置与超参数 (在这里修改你的毕设参数)
# =================================================================
# --- V-REP 场景对象名称 ---
NAME_L_P2 = "UR10_L_p2"
NAME_R_P2 = "UR10_R_p2"
NAME_PART = "Part1_Reference"
NAME_L_TARGET = "/UR10_L_target"
NAME_R_TARGET = "/UR10_R_target"
NAME_L_SENSOR = "UR10_L_forceSensor"
NAME_R_SENSOR = "UR10_R_forceSensor"
PREFIX_L_JOINT = "JL"
PREFIX_R_JOINT = "JR"

# --- 任务目标 ---
PART_GOAL_POS = np.array([0.0, -0.775, 0.220])
PART_GOAL_ORI = np.array([0, 0, 0])
PART_INIT_ORI_EULER = np.array([0, 0, 0])

# --- 仿真时间参数 ---
DURATION = 10.0       # 单次 Episode 总时长 (秒)
STEP_SIZE = 0.05      # 仿真步长 (秒)
MAX_STEPS = int(DURATION / STEP_SIZE)

# --- 动力学物理参数 (Python 虚拟计算用) ---
OBJ_MASS = 2.0        # 物体质量 (kg)
GRAVITY = np.array([0, 0, -9.81])

# --- 变阻抗基准值 (RL 将在此基础上调整) ---
# 你的目标是输出 MBK，这里定义基准值
BASE_M = 5.0          # 基准虚拟质量
BASE_K = 800.0        # 基准虚拟刚度 (N/m)
BASE_B = 100.0        # 基准虚拟阻尼 (Ns/m)

# 动作映射范围: 网络输出[-1, 1] -> 映射到 [MIN, MAX] 倍率
ACTION_SCALE_MIN = 0.1  # 最软/最轻
ACTION_SCALE_MAX = 5.0  # 最硬/最重

# --- 机械臂动力学参数 (UR10) ---
UR10_LINK_MASSES = [7.1, 12.7, 4.27, 2.0, 2.0, 0.365]
UR10_LINK_COMS = [
    [0, -0.02, 0], [0.3, 0, 0.1], [0.25, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0]
]
# 关节力矩安全上限 (Nm) - 用于归一化 Reward
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

# --- 奖励函数权重配置 ---
# Reward = REWARD_STEP - (W_POS * PosErr + W_TORQUE * TorquePenalty)
REWARD_STEP = 1.0       # [新增] 每步的基础生存奖励
REWARD_W_POS = 20.0     # 位置误差惩罚权重 (主项)
REWARD_W_TORQUE = 0.02  # 力矩过大惩罚权重 (次项，鼓励柔顺)
REWARD_SUCCESS = 100.0  # 成功奖励
REWARD_FAIL = -100.0    # 失败/发散惩罚

# --- 强化学习训练参数 ---
STATE_DIM = 9   # [PosErr(3), Velocity(3), Force(3)]
ACTION_DIM = 3  # [M_scale, K_scale, B_scale] -> 输出 MBK
HIDDEN_DIM = 256
BATCH_SIZE = 128
BUFFER_SIZE = 50000
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 结果保存路径 ---
RESULT_FILE = "training_rewards.txt" # 只保留这个文件
MODEL_DIR = "./saved_models"
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

# =================================================================
# 2. 数学模型 (用于计算 12 个关节的动力学力矩)
# =================================================================
class UR10_MathModel:
    def __init__(self, base_offset_x):
        # Modified DH Parameters for UR10
        self.dh = [
            [0, 0, 0.1273, 0], [np.pi/2, 0, 0, 0], [0, -0.612, 0, 0],
            [0, -0.5723, 0, 0], [np.pi/2, 0, 0.1639, 0], [-np.pi/2, 0, 0.1157, 0]
        ]
        self.base_pos = np.array([base_offset_x, 0, 0])
        self.masses = UR10_LINK_MASSES
        self.coms = UR10_LINK_COMS

    def forward_kine(self, q):
        """正运动学计算"""
        T = np.eye(4); T[:3,3] = self.base_pos
        transforms = [T]
        for i, (alp, a, d, off) in enumerate(self.dh):
            theta = q[i] + off
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alp), np.sin(alp)
            Ti = np.array([[ct, -st, 0, a], [st*ca, ct*ca, -sa, -sa*d], [st*sa, ct*sa, ca, ca*d], [0,0,0,1]])
            T = T @ Ti
            transforms.append(T)
        return T, transforms

    def get_jacobian(self, q):
        """数值法计算雅可比矩阵"""
        n = 6; eps = 1e-4; J = np.zeros((6, n))
        T0, _ = self.forward_kine(q)
        p0 = T0[:3,3]
        for i in range(n):
            q_ = np.array(q); q_[i] += eps
            T1, _ = self.forward_kine(q_)
            J[:3,i] = (T1[:3,3] - p0)/eps
        return J

    def compute_gravity_torques(self, q):
        """重力补偿力矩计算"""
        _, Ts = self.forward_kine(q)
        tau = np.zeros(6)
        p_coms = []
        for i in range(6):
            p_local = np.append(self.coms[i], 1)
            p_coms.append((Ts[i+1] @ p_local)[:3])
        z_axes = [Ts[i][:3, 2] for i in range(6)]
        p_joints = [Ts[i][:3, 3] for i in range(6)]
        for i in range(5, -1, -1):
            t_i = 0
            for j in range(i, 6):
                f_g = self.masses[j] * (-GRAVITY) 
                r = p_coms[j] - p_joints[i]
                t_i += np.dot(np.cross(r, f_g), z_axes[i])
            tau[i] = t_i
        return tau

    def compute_total_torque(self, q, f_ext):
        """计算总关节力矩: 重力项 + 雅可比转置映射的末端外力"""
        tau_g = self.compute_gravity_torques(q)
        J = self.get_jacobian(q)
        wrench = np.concatenate([f_ext, [0,0,0]])
        tau_ext = J.T @ wrench
        return tau_g + tau_ext

# =================================================================
# 3. 虚拟动力学引擎 (变阻抗核心)
# =================================================================
class VirtualPhysicsEngine:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.g_force = OBJ_MASS * GRAVITY 

    def reset(self, p):
        self.pos = np.array(p)
        self.vel = np.zeros(3)

    def step(self, target_pos, m_scale, k_scale, b_scale, f_disturb=np.zeros(3)):
        # [核心] 实时变阻抗: MBK 均可调
        def map_scale(val):
            norm = (val + 1.0) / 2.0 
            return norm * (ACTION_SCALE_MAX - ACTION_SCALE_MIN) + ACTION_SCALE_MIN

        # 计算实际物理参数
        M = BASE_M * map_scale(m_scale)
        K = np.diag([BASE_K * map_scale(k_scale)] * 3)
        B = np.diag([BASE_B * map_scale(b_scale)] * 3)
        
        # 动力学计算 (F = Ma = K*err - B*vel + G + Disturbance)
        err = target_pos - self.pos
        f_compliance = K @ err - B @ self.vel
        f_net = f_compliance + self.g_force + f_disturb
        
        acc = f_net / M 
        self.vel += acc * STEP_SIZE
        self.pos += self.vel * STEP_SIZE
        
        load_on_arm = -f_compliance 
        return self.pos, load_on_arm

class TrajectoryPlanner:
    def __init__(self, start_pos, start_quat, goal_pos, goal_euler):
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        # Slerp 插值用于姿态平滑过渡
        self.slerp = Slerp([0, 1], R.from_matrix([
            R.from_quat(start_quat).as_matrix(), 
            R.from_euler('xyz', goal_euler).as_matrix()
        ]))

    def get_ideal_pose(self, t_ratio):
        t = min(max(t_ratio, 0.0), 1.0)
        pos = (1 - t) * self.start_pos + t * self.goal_pos
        mat = np.eye(4)
        mat[:3, 3] = pos
        mat[:3, :3] = self.slerp([t]).as_matrix()[0]
        return mat

# =================================================================
# 4. RL 环境封装 (Gym Style)
# =================================================================
class ImpedanceEnv:
    def __init__(self):
        # 初始化 V-REP 连接
        sim.simxFinish(-1)
        self.cid = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        if self.cid == -1: raise Exception("V-REP 连接失败")
        sim.simxSynchronous(self.cid, True)
        sim.simxStartSimulation(self.cid, sim.simx_opmode_blocking)

        # 获取句柄
        self.h_part = self._get_h(NAME_PART)
        self.h_l_tgt = self._get_h(NAME_L_TARGET)
        self.h_r_tgt = self._get_h(NAME_R_TARGET)
        self.h_lp2 = self._get_h(NAME_L_P2)
        self.h_rp2 = self._get_h(NAME_R_P2)
        
        self.hj_l = [self._get_h(f"{PREFIX_L_JOINT}{i}") for i in range(1,7)]
        self.hj_r = [self._get_h(f"{PREFIX_R_JOINT}{i}") for i in range(1,7)]

        # [新增] 读取初始关节角度 (在训练开始前)
        print("正在读取初始机械臂姿态...")
        self.init_q_l = self._get_q(self.hj_l)
        self.init_q_r = self._get_q(self.hj_r)
        print("初始姿态已保存.")

        # 初始化模型
        self.robot_l = UR10_MathModel(-0.525)
        self.robot_r = UR10_MathModel(0.525)
        self.physics = VirtualPhysicsEngine()
        
        self.m_start, self.off_l, self.off_r = self._init_alignment()
        start_quat = R.from_matrix(self.m_start[:3,:3]).as_quat()
        self.planner = TrajectoryPlanner(self.m_start[:3,3], start_quat, PART_GOAL_POS, PART_GOAL_ORI)
        
        self.current_step = 0

    def _get_h(self, n): return sim.simxGetObjectHandle(self.cid, n, sim.simx_opmode_blocking)[1]
    
    def _get_mat(self, h):
        _, p = sim.simxGetObjectPosition(self.cid, h, -1, sim.simx_opmode_blocking)
        _, q = sim.simxGetObjectQuaternion(self.cid, h, -1, sim.simx_opmode_blocking)
        m = np.eye(4); m[:3,3] = p; m[:3,:3] = R.from_quat(q).as_matrix()
        return m

    def _set_mat(self, h, m):
        p = m[:3,3]; q = R.from_matrix(m[:3,:3]).as_quat()
        sim.simxSetObjectPosition(self.cid, h, -1, p.tolist(), sim.simx_opmode_oneshot)
        sim.simxSetObjectQuaternion(self.cid, h, -1, q.tolist(), sim.simx_opmode_oneshot)

    def _get_q(self, hs):
        return [sim.simxGetJointPosition(self.cid, h, sim.simx_opmode_blocking)[1] for h in hs]

    def _set_q(self, hs, qs):
        """[新增] 设置一组关节的角度"""
        for h, q in zip(hs, qs):
            sim.simxSetJointPosition(self.cid, h, q, sim.simx_opmode_oneshot)

    def _init_alignment(self):
        m_l, m_r = self._get_mat(self.h_lp2), self._get_mat(self.h_rp2)
        init_p = (m_l[:3,3] + m_r[:3,3])/2
        m_start = np.eye(4); m_start[:3,3] = init_p
        m_start[:3,:3] = R.from_euler('xyz', PART_INIT_ORI_EULER).as_matrix()
        
        self._set_mat(self.h_part, m_start)
        sim.simxSynchronousTrigger(self.cid)
        
        inv = np.linalg.inv(m_start)
        return m_start, inv @ m_l, inv @ m_r

    def reset(self):
        self.current_step = 0
        self.physics.reset(self.m_start[:3,3])
        
        # 恢复物体和目标点位置
        self._set_mat(self.h_part, self.m_start)
        self._set_mat(self.h_l_tgt, self.m_start @ self.off_l)
        self._set_mat(self.h_r_tgt, self.m_start @ self.off_r)
        
        # [新增] 强制复位关节角度到初始状态
        self._set_q(self.hj_l, self.init_q_l)
        self._set_q(self.hj_r, self.init_q_r)
        
        sim.simxSynchronousTrigger(self.cid)
        
        return np.zeros(STATE_DIM)

    def step(self, action):
        self.current_step += 1
        t_ratio = self.current_step / MAX_STEPS
        
        m_ideal = self.planner.get_ideal_pose(t_ratio)
        ideal_pos = m_ideal[:3, 3]
        
        # 扰动注入 (随机化)
        f_disturb = np.zeros(3)
        if self.current_step == 100: 
            f_disturb = np.array([0, np.random.uniform(30, 80), 0])
        
        # 动力学演化 (Action: M_scale, K_scale, B_scale)
        real_pos, load_force = self.physics.step(ideal_pos, action[0], action[1], action[2], f_disturb)
        
        # V-REP 更新: 将计算出的位置强行设置给物体，机械臂通过IK或约束跟随
        m_curr = m_ideal.copy(); m_curr[:3,3] = real_pos
        self._set_mat(self.h_part, m_curr)
        self._set_mat(self.h_l_tgt, m_curr @ self.off_l)
        self._set_mat(self.h_r_tgt, m_curr @ self.off_r)
        sim.simxSynchronousTrigger(self.cid)
        
        # 力矩计算 (用于 Reward)
        q_l, q_r = self._get_q(self.hj_l), self._get_q(self.hj_r)
        half_load = load_force / 2.0 
        tau_l = self.robot_l.compute_total_torque(q_l, half_load)
        tau_r = self.robot_r.compute_total_torque(q_r, half_load)
        
        # 奖励计算
        pos_err = np.linalg.norm(ideal_pos - real_pos)
        # 惩罚力矩过大 (鼓励柔顺)
        tau_penalty = np.mean(np.abs(tau_l) / TORQUE_LIMITS) + np.mean(np.abs(tau_r) / TORQUE_LIMITS)
        
        # [修改] 增加每步的基础奖励 REWARD_STEP (+1)
        reward = REWARD_STEP - REWARD_W_POS * pos_err - REWARD_W_TORQUE * tau_penalty
        
        done = False
        # [修改] 失败判定：误差过大 (>0.2) 视为“不能解决/发散”，强制给予 -100 惩罚并结束
        if pos_err > 0.2: 
            reward = REWARD_FAIL # -100
            done = True
        elif self.current_step >= MAX_STEPS:
            done = True
            if pos_err < 0.02: reward += REWARD_SUCCESS
            
        state = np.concatenate([ideal_pos - real_pos, self.physics.vel, load_force])
        return state, reward, done, pos_err, tau_l, tau_r

# =================================================================
# 5. DDPG Agent
# =================================================================
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, a_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
    def forward(self, s, a): return self.net(torch.cat([s, a], 1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        s, a, r, ns, d = zip(*random.sample(self.buffer, batch_size))
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)
    def __len__(self): return len(self.buffer)

# =================================================================
# 6. 训练主循环
# =================================================================
def train():
    try:
        env = ImpedanceEnv()
    except Exception as e:
        print(f"Error: {e}")
        print("请检查: 1. V-REP是否开启 2. sim.py/remoteApi.dll 是否在当前目录")
        return

    actor = Actor(STATE_DIM, ACTION_DIM).to(DEVICE)
    actor_target = Actor(STATE_DIM, ACTION_DIM).to(DEVICE)
    actor_target.load_state_dict(actor.state_dict())
    
    critic = Critic(STATE_DIM, ACTION_DIM).to(DEVICE)
    critic_target = Critic(STATE_DIM, ACTION_DIM).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    
    a_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    c_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    # 清空旧的结果文件
    if os.path.exists(RESULT_FILE): os.remove(RESULT_FILE)
    
    print(f"\n>>> 开始训练 | Device: {DEVICE} | 目标: 优化 MBK")
    print(f">>> 结果将保存至: {RESULT_FILE}")

    try:
        reward_history = []
        for ep in range(200):
            state = env.reset()
            ep_reward = 0
            noise = 0.2 * (0.995 ** ep)
            
            for step in range(MAX_STEPS):
                s_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action = actor(s_tensor).cpu().numpy()[0]
                action = np.clip(action + np.random.normal(0, noise, ACTION_DIM), -1, 1)
                
                # 交互
                next_state, r, done, err, tau_l, tau_r = env.step(action)
                buffer.push(state, action, r, next_state, done)
                
                # 训练
                if len(buffer) > BATCH_SIZE:
                    s, a, rw, ns, d = buffer.sample(BATCH_SIZE)
                    s = torch.FloatTensor(s).to(DEVICE)
                    a = torch.FloatTensor(a).to(DEVICE)
                    rw = torch.FloatTensor(rw).unsqueeze(1).to(DEVICE)
                    ns = torch.FloatTensor(ns).to(DEVICE)
                    d = torch.FloatTensor(d).unsqueeze(1).to(DEVICE)
                    
                    with torch.no_grad():
                        target_q = rw + (1 - d) * GAMMA * critic_target(ns, actor_target(ns))
                    c_loss = nn.MSELoss()(critic(s, a), target_q)
                    c_opt.zero_grad(); c_loss.backward(); c_opt.step()
                    
                    a_loss = -critic(s, actor(s)).mean()
                    a_opt.zero_grad(); a_loss.backward(); a_opt.step()
                    
                    for p, tp in zip(actor.parameters(), actor_target.parameters()):
                        tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
                    for p, tp in zip(critic.parameters(), critic_target.parameters()):
                        tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
                
                state = next_state
                ep_reward += r
                
                # [实时看板] 覆盖式打印
                tau_l_str = ",".join([f"{t:.1f}" for t in tau_l])
                tau_r_str = ",".join([f"{t:.1f}" for t in tau_r])
                msg = (f"\rEp {ep} St {step} | Err:{err*1000:4.1f}mm | R:{r:.2f} | "
                       f"L_Tau:[{tau_l_str}] | R_Tau:[{tau_r_str}]")
                sys.stdout.write(msg[:150].ljust(150))
                sys.stdout.flush()
                
                if done: break
            
            # [本轮结束]
            sys.stdout.write("\r" + " "*150 + "\r") # 清行
            print(f"Episode {ep} Done. Total Reward: {ep_reward:.2f}")
            
            with open(RESULT_FILE, "a") as f:
                f.write(f"{ep},{ep_reward}\n")
            
            reward_history.append(ep_reward)
            
            # [新增] 每轮训练结束后，停止一秒
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n>>> 训练被手动停止")
    finally:
        # 结束时绘图
        if reward_history:
            plt.figure(figsize=(10, 5))
            plt.plot(reward_history)
            plt.title("Training Reward Curve")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.grid(True)
            plt.show()
            
        sim.simxStopSimulation(env.cid, sim.simx_opmode_blocking)
        sim.simxFinish(env.cid)

if __name__ == "__main__":
    train()