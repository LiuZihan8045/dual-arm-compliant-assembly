import sim
import sys
import time
import numpy as np

# ================= 配置区域 =================
# 场景中的路径点名称 (你要去哪里)
NAME_L_P1 = "UR10_L_p2"
NAME_R_P1 = "UR10_R_p2"

# IK 目标名称 (Lua脚本正在追踪的那个Dummy)
NAME_L_TARGET = "/UR10_L_target"
NAME_R_TARGET = "/UR10_R_target"

# 关节前缀 (JL1...JL6, JR1...JR6)
PREFIX_L = "JL"
PREFIX_R = "JR"
# ===========================================

def connect_vrep():
    print("正在连接 V-REP...")
    sim.simxFinish(-1)
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if client_id == -1:
        print("连接失败！")
        sys.exit()
    return client_id

def copy_pose(client_id, source_name, target_name):
    """
    将 source_name 的位姿完全复制给 target_name
    """
    # 1. 获取句柄
    _, h_src = sim.simxGetObjectHandle(client_id, source_name, sim.simx_opmode_blocking)
    _, h_tgt = sim.simxGetObjectHandle(client_id, target_name, sim.simx_opmode_blocking)
    
    if h_src == 0 or h_tgt == 0:
        print(f"Error: 找不到 {source_name} 或 {target_name}")
        return

    # 2. 读取源的位置和姿态 (相对于世界坐标系 -1)
    _, pos = sim.simxGetObjectPosition(client_id, h_src, -1, sim.simx_opmode_blocking)
    _, ori = sim.simxGetObjectOrientation(client_id, h_src, -1, sim.simx_opmode_blocking)
    
    # 3. 设置给目标
    sim.simxSetObjectPosition(client_id, h_tgt, -1, pos, sim.simx_opmode_oneshot)
    sim.simxSetObjectOrientation(client_id, h_tgt, -1, ori, sim.simx_opmode_oneshot)
    
    print(f"已将 {target_name} 移动到 {source_name}")

def get_and_print_joints(client_id, arm_name, prefix):
    """
    读取并打印 6 个关节角度
    """
    print(f"\n--- {arm_name} ({prefix}1-{prefix}6) 当前关节角度 ---")
    print(f"{'关节':<10} | {'角度(度)':<15} | {'弧度(rad)':<15}")
    print("-" * 45)
    
    for i in range(1, 7):
        joint_name = f"{prefix}{i}" # 拼接关节名，如 JL1
        
        # 获取句柄
        _, h_joint = sim.simxGetObjectHandle(client_id, joint_name, sim.simx_opmode_blocking)
        
        # 读取角度
        res, angle_rad = sim.simxGetJointPosition(client_id, h_joint, sim.simx_opmode_blocking)
        
        if res == sim.simx_return_ok:
            angle_deg = np.degrees(angle_rad)
            print(f"{joint_name:<10} | {angle_deg:<15.4f} | {angle_rad:<15.4f}")
        else:
            print(f"{joint_name:<10} | 读取失败")

def main():
    client_id = connect_vrep()
    
    # 开启同步模式，确保我们能控制时间
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)
    
    try:
        # 1. 直接将 Target 移动到 P1 的位置
        # 这样 V-REP 的 IK 算法会立刻把机械臂拉过去
        copy_pose(client_id, NAME_L_P1, NAME_L_TARGET)
        copy_pose(client_id, NAME_R_P1, NAME_R_TARGET)
        
        # 2. 触发仿真步进，等待 IK 解算稳定
        # 即使是瞬移 Target，IK 解算和物理引擎也需要几帧的时间来对齐关节
        print("等待机械臂稳定到位...")
        for _ in range(20): # 等待约 1秒 (20 * 0.05s)
            sim.simxSynchronousTrigger(client_id)
            
        # 3. 读取此时的关节角度
        get_and_print_joints(client_id, "左臂", PREFIX_L)
        get_and_print_joints(client_id, "右臂", PREFIX_R)
        
        print("\n读取完成。")
        
        # 保持一会儿看看效果
        for _ in range(20): sim.simxSynchronousTrigger(client_id)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)
        sim.simxFinish(client_id)

if __name__ == "__main__":
    main()