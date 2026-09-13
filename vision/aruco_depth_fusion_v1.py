# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import numpy as np
import cv2
import cv2.aruco as aruco
from openni import openni2
from openni import _openni2 as c_api

# 防止库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. 用户配置区 (关键！)
# ==========================================
# Aruco 码实际边长 (单位: 米)
MARKER_SIZE = 0.05 

# 相机内参 (Astra Pro RGB 典型值，用于 6D 解算)
fx, fy = 580.0, 580.0
cx, cy_opt = 320.0, 240.0
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy_opt], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# ==========================================
# 2. 辅助函数
# ==========================================
def rotation_matrix_to_euler_angles(R):
    """旋转矩阵转欧拉角 (度)"""
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

def get_robust_depth(depth_frame, x, y, roi_size=2):
    """获取中心点周围区域的深度中位数，防止噪点"""
    h, w = depth_frame.shape
    x_min = max(0, x - roi_size)
    x_max = min(w, x + roi_size + 1)
    y_min = max(0, y - roi_size)
    y_max = min(h, y + roi_size + 1)
    
    roi = depth_frame[y_min:y_max, x_min:x_max]
    valid_depths = roi[roi > 0]
    
    if len(valid_depths) == 0: return 0
    return np.median(valid_depths)

# ==========================================
# 3. 系统初始化 (安全启动模式)
# ==========================================
print("--- 正在启动双目融合系统 ---")

# --- Step A: 启动 RGB 相机 ---
print("1. 正在打开 RGB 相机...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("⚠️ 尝试端口 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ RGB 相机无法启动")
        sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("✅ RGB 就绪")

# 给 USB 总线一点喘息时间
time.sleep(1.0)

# --- Step B: 启动 深度 相机 ---
print("2. 正在打开深度相机 (OpenNI)...")
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    openni2.initialize(script_dir)
    dev = openni2.Device.open_any()
    
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    # 设置深度模式: 1mm 精度
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=100, resolutionX=640, resolutionY=480, fps=30))
    
    # 【关键】开启硬件对齐：让深度的 (u,v) 和 RGB 的 (u,v) 重合
    if dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        print("✅ 深度-彩色 对齐已开启")
    else:
        print("⚠️ 警告：无法开启硬件对齐，深度数据可能偏移")
        
    print("✅ 深度引擎就绪")
except Exception as e:
    print(f"❌ OpenNI 启动失败: {e}\n请运行 kill_camera.py 后重试。")
    cap.release()
    sys.exit()

# Aruco 配置
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# ==========================================
# 4. 主循环
# ==========================================
print("\n=== 系统运行中 (按 'q' 退出) ===")
print("绿色文字 = Aruco 算出的坐标 (基于图像几何)")
print("黄色文字 = 深度相机测出的距离 (基于红外物理测量)")

try:
    while True:
        # 1. 读取 RGB
        ret, frame = cap.read()
        if not ret: continue
        
        # 2. 读取深度
        d_frame = depth_stream.read_frame()
        d_frame_data = d_frame.get_buffer_as_uint16()
        d_img = np.frombuffer(d_frame_data, dtype=np.uint16).reshape(480, 640)

        # 3. Aruco 检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # PnP 解算
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                # --- A: 获取 6D 信息 (Aruco) ---
                rvec, tvec = rvecs[i][0], tvecs[i][0]
                
                # 计算中心点
                corner = corners[i][0]
                cx = int((corner[0][0] + corner[2][0]) / 2)
                cy = int((corner[0][1] + corner[2][1]) / 2)

                # 欧拉角
                R_mat, _ = cv2.Rodrigues(rvec)
                roll, pitch, yaw = rotation_matrix_to_euler_angles(R_mat)

                # Aruco 算出的坐标 (mm)
                aruco_x = tvec[0] * 1000
                aruco_y = tvec[1] * 1000
                aruco_z = tvec[2] * 1000

                # --- B: 获取 深度信息 (OpenNI) ---
                # 在中心点 (cx, cy) 采样深度
                depth_z = get_robust_depth(d_img, cx, cy)
                
                # --- 绘图与显示 ---
                # 1. 画坐标轴
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                # 2. 画边框
                aruco.drawDetectedMarkers(frame, corners)
                # 3. 画中心点
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # 4. 信息显示
                # 第一行：Aruco 算出的 6D (绿色)
                text_aruco = f"Aruco Pose: X:{aruco_x:.0f} Y:{aruco_y:.0f} Z:{aruco_z:.0f} | RPY:{roll:.0f},{pitch:.0f},{yaw:.0f}"
                # 第二行：深度相机测出的 Z (黄色)
                text_depth = f"Depth Sensor Z: {depth_z} mm"
                
                # 显示位置调整
                cv2.putText(frame, text_aruco, (cx - 80, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, text_depth, (cx - 80, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 控制台打印对比
                print(f"ID:{ids[i][0]} | Aruco Z: {aruco_z:.1f}mm vs Depth Z: {depth_z}mm")

        cv2.imshow("Astra Pro 6D Fusion", frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    print("正在关闭设备...")
    depth_stream.stop()
    openni2.unload()
    cap.release()
    cv2.destroyAllWindows()