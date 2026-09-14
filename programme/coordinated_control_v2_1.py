import sim  # 也就是 vrep.py 或 sim.py
import sys
import time
import numpy as np

# ================= 参数设置 =================
STEP_SIZE = 0.05      # 仿真步长 (秒)
DURATION = 20.0       # 总运动时长 (秒)
TOTAL_STEPS = int(DURATION / STEP_SIZE)

# 涉及的对象名称
NAME_START = "UR10_R_start"   # 场景里已有的起点
NAME_END   = "UR10_R_p1"      # 场景里已有的终点
NAME_TARGET= "/UR10_R_target" # 要移动的目标
# ===========================================

def connect_vrep():
    print("正在连接 V-REP (UR10 控制)...")
    sim.simxFinish(-1) 
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if client_id != -1:
        print("连接成功!")
        return client_id
    else:
        print("连接失败!")
        sys.exit()

def get_pose(client_id, object_name):
    """
    获取对象的 [x, y, z] 和 [alpha, beta, gamma]
    """
    # 1. 获取句柄
    res, handle = sim.simxGetObjectHandle(client_id, object_name, sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print(f"错误: 找不到对象 '{object_name}'")
        return None, None, None

    # 2. 获取位置 (相对于世界坐标系 -1)
    res_pos, pos = sim.simxGetObjectPosition(client_id, handle, -1, sim.simx_opmode_blocking)
    
    # 3. 获取姿态 (相对于世界坐标系 -1)
    res_ori, ori = sim.simxGetObjectOrientation(client_id, handle, -1, sim.simx_opmode_blocking)
    
    if res_pos == sim.simx_return_ok and res_ori == sim.simx_return_ok:
        return handle, np.array(pos), np.array(ori)
    else:
        print(f"错误: 无法读取 '{object_name}' 的位姿数据")
        return None, None, None

def main():
    client_id = connect_vrep()

    # 1. 开启同步模式 (关键：保证时间步长精确为 0.05s)
    sim.simxSynchronous(client_id, True)
    sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)

    try:
        print("=== 读取路径点信息 ===")
        
        # 获取 Start 点信息
        _, start_pos, start_ori = get_pose(client_id, NAME_START)
        # 获取 P1 点信息
        _, end_pos, end_ori = get_pose(client_id, NAME_END)
        # 获取 Target 句柄 (我们需要控制它)
        res, target_handle = sim.simxGetObjectHandle(client_id, NAME_TARGET, sim.simx_opmode_blocking)

        if start_pos is None or end_pos is None or res != sim.simx_return_ok:
            print("无法获取必要的路径点或句柄，程序退出。")
            return

        print(f"起点: {start_pos} | 姿态: {start_ori}")
        print(f"终点: {end_pos} | 姿态: {end_ori}")
        print(f"计划: {DURATION}秒内完成运动，共 {TOTAL_STEPS} 步。")
        
        # 预热物理引擎
        for _ in range(10):
            sim.simxSynchronousTrigger(client_id)

        print("=== 开始插值运动 ===")
        
        start_time = time.time()

        # 循环插值
        for i in range(TOTAL_STEPS + 1):
            # 计算进度 t (0.0 到 1.0)
            t = i / TOTAL_STEPS
            
            # --- 线性插值 (Lerp) ---
            # 位置插值: P = P_start + t * (P_end - P_start)
            current_pos = (1 - t) * start_pos + t * end_pos
            
            # 姿态插值: O = O_start + t * (O_end - O_start)
            # 注意：简单的欧拉角线性插值在角度变化很大(>180度)时可能不理想，
            # 但对于一般的路径点移动是足够的。
            current_ori = (1 - t) * start_ori + t * end_ori
            
            # --- 发送命令给 V-REP ---
            sim.simxSetObjectPosition(client_id, target_handle, -1, current_pos.tolist(), sim.simx_opmode_oneshot)
            sim.simxSetObjectOrientation(client_id, target_handle, -1, current_ori.tolist(), sim.simx_opmode_oneshot)
            
            # --- 触发下一步 ---
            # 这会让 V-REP 前进 0.05 秒 (前提是 V-REP 里的 dt 设置也是 0.05)
            sim.simxSynchronousTrigger(client_id)
            
            # 可选：打印进度
            if i % 20 == 0:
                print(f"进度: {t*100:.1f}%")

        real_duration = time.time() - start_time
        print(f"运动结束。实际耗时(含通信开销): {real_duration:.2f}秒")
        print("保持 5 秒后退出...")
        
        for _ in range(100): # 保持一会
            sim.simxSynchronousTrigger(client_id)

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)
        sim.simxFinish(client_id)
        print("连接关闭")

if __name__ == "__main__":
    main()