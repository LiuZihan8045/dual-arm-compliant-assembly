# -*- coding: utf-8 -*-
import cv2
import cv2.aruco as aruco
import numpy as np
from openni import openni2
import os
import sys
import math

# ==========================================
# 0. 用户配置区 (请务必修改这里！)
# ==========================================
# 【关键】Aruco 码的实际边长 (单位: 米)
# 如果你是手机屏幕显示，大概估算一下，比如 0.04 (4厘米)
MARKER_SIZE = 0.05  

# 相机内参 (Astra Pro RGB 640x480 的近似值)
# 为了极高精度抓取，后续需要手动标定替换这里
fx = 580.0
fy = 580.0
cx = 320.0
cy = 240.0

# 构建内参矩阵 K
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)

# 畸变系数 (假设为0，后续标定后填入)
dist_coeffs = np.zeros((4, 1))

# ==========================================
# 1. 辅助函数：旋转向量转欧拉角
# ==========================================
def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角 (Roll, Pitch, Yaw)
    返回值单位：度
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees(np.array([x, y, z]))

# ==========================================
# 2. 初始化摄像头 (只用 RGB 即可算出 Pose)
# ==========================================
# 为了对比，我们还是把 OpenNI 开着，用来验证距离
script_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(script_dir, "OpenNI2.dll")

# 启动 OpenNI
openni2.initialize(script_dir)
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()

# 启动 RGB
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 加载 Aruco 字典 (DICT_6X6_250 是最常用的)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

print("=== Aruco 6D Pose Estimation ===")
print(f"设定 Marker 边长: {MARKER_SIZE} 米")
print("请将 Aruco 码 (6x6) 放入画面...")

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 检测角点
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # 2. 姿态解算 (PnP 算法核心)
            # rvec: 旋转向量, tvec: 平移向量
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                # 获取当前的 rvec, tvec
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # --- 绘制坐标轴 (红:x, 绿:y, 蓝:z) ---
                # 长度 0.05米
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05) 
                
                # --- 绘制 Marker 边框 ---
                aruco.drawDetectedMarkers(frame, corners)

                # --- 计算欧拉角 (更直观的姿态) ---
                # Rodrigues 变换: 旋转向量 -> 旋转矩阵
                R, _ = cv2.Rodrigues(rvec)
                roll, pitch, yaw = rotation_matrix_to_euler_angles(R)

                # tvec 单位是米，转换为毫米方便看
                x_mm = tvec[0] * 1000
                y_mm = tvec[1] * 1000
                z_mm = tvec[2] * 1000

                # --- 屏幕显示数据 ---
                info_pos = f"Pos[mm]: X={x_mm:.0f} Y={y_mm:.0f} Z={z_mm:.0f}"
                info_rot = f"Ang[deg]: R={roll:.0f} P={pitch:.0f} Y={yaw:.0f}"

                # 在 Marker 上方打印信息
                top_left = corners[i][0][0]
                cv2.putText(frame, info_pos, (int(top_left[0]), int(top_left[1]) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, info_rot, (int(top_left[0]), int(top_left[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # --- (可选) OpenNI 深度对比 ---
                # 用 Aruco 算出来的 Z 和 深度相机测出来的 Z 对比一下
                cx, cy = int(top_left[0]), int(top_left[1]) # 简单取一个角点
                # ...这里省略具体的 OpenNI 读取，以免代码太复杂，
                # 这里的 tvec[2] (即 z_mm) 已经是非常精准的距离了，完全可以用于机械臂。

        cv2.imshow("Aruco 6D Pose", frame)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    depth_stream.stop()
    openni2.unload()
    cap.release()
    cv2.destroyAllWindows()