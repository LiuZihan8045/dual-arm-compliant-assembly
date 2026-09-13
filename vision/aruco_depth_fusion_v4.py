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
# 1. 用户配置区 (请根据实际情况调整！)
# ==========================================
# 【核心开关】如果热力图方向反了，请修改这里为 True 或 False
NEED_MIRROR_DEPTH = False

# Aruco 边长 (米)
MARKER_SIZE = 0.05 

# 相机内参
fx, fy = 580.0, 580.0
cx, cy_opt = 320.0, 240.0
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy_opt], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# ==========================================
# 2. 辅助函数
# ==========================================
def get_robust_depth(depth_frame, x, y, roi_size=2):
    """获取区域深度中位数"""
    h, w = depth_frame.shape
    if x < 0 or x >= w or y < 0 or y >= h: return 0
    x_min = max(0, x - roi_size)
    x_max = min(w, x + roi_size + 1)
    y_min = max(0, y - roi_size)
    y_max = min(h, y + roi_size + 1)
    roi = depth_frame[y_min:y_max, x_min:x_max]
    valid_depths = roi[roi > 0]
    if len(valid_depths) == 0: return 0
    return np.median(valid_depths)

# ==========================================
# 3. 初始化
# ==========================================
print(f"--- 启动系统 (镜像开关: {NEED_MIRROR_DEPTH}) ---")

# A. 启动 RGB
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): sys.exit("❌ RGB 无法启动")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("✅ RGB 就绪")
time.sleep(0.5)

# B. 启动 OpenNI
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    openni2.initialize(script_dir)
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=100, resolutionX=640, resolutionY=480, fps=30))
    
    # 尝试硬件对齐
    if dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        
    # 尝试在驱动层关闭镜像
    try:
        depth_stream.set_mirroring_enabled(False)
    except:
        pass
        
    print("✅ 深度引擎就绪")
except Exception as e:
    print(f"❌ OpenNI 错误: {e}")
    sys.exit()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# ==========================================
# 4. 主循环
# ==========================================
print("\n=== 按 'q' 退出 ===")
print("请确认左右画面的红点是否均指向物体中心")

try:
    while True:
        # 1. 读取 RGB
        ret, frame_rgb = cap.read()
        if not ret: continue

        # 2. 读取深度 (带开关控制)
        try:
            d_frame = depth_stream.read_frame()
            if d_frame.dataSize != 614400: continue
            
            d_frame_data = d_frame.get_buffer_as_uint16()
            d_img_raw = np.frombuffer(d_frame_data, dtype=np.uint16).reshape(480, 640)
            
            # 【关键修改】根据开关决定是否镜像
            if NEED_MIRROR_DEPTH:
                d_img_raw = cv2.flip(d_img_raw, 1)
            
            depth_valid = True
        except:
            depth_valid = False
            d_img_raw = np.zeros((480, 640), dtype=np.uint16)

        # 3. 生成深度热力图
        if depth_valid:
            depth_vis = cv2.normalize(d_img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        else:
            depth_vis_color = np.zeros((480, 640, 3), dtype=np.uint8)

        # 4. Aruco 检测
        if depth_valid:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                
                for i in range(len(ids)):
                    # 获取数据
                    rvec, tvec = rvecs[i][0], tvecs[i][0]
                    c = corners[i][0]
                    cx, cy = int((c[0][0] + c[2][0]) / 2), int((c[0][1] + c[2][1]) / 2)
                    
                    # 深度采样 (直接在处理好的 d_img_raw 上取值)
                    depth_z = get_robust_depth(d_img_raw, cx, cy)
                    aruco_z = tvec[2] * 1000 
                    
                    # --- 绘图 (RGB) ---
                    cv2.drawFrameAxes(frame_rgb, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    aruco.drawDetectedMarkers(frame_rgb, corners)
                    
                    # 红点 + 十字准星 (更显眼)
                    cv2.circle(frame_rgb, (cx, cy), 5, (0, 0, 255), -1) 
                    cv2.drawMarker(frame_rgb, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
                    
                    # 文字
                    cv2.putText(frame_rgb, f"Aruco: {aruco_z:.0f}", (cx-60, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame_rgb, f"Depth: {depth_z}", (cx-60, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # --- 绘图 (Depth Heatmap) ---
                    # 这里的坐标 (cx, cy) 必须和 RGB 图完全一致
                    cv2.circle(depth_vis_color, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.drawMarker(depth_vis_color, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
                    cv2.putText(depth_vis_color, f"{depth_z}mm", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 5. 拼接与显示
        combined = np.hstack((frame_rgb, depth_vis_color))
        cv2.imshow("Astra Pro Mirror Fixed", combined)

        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"❌ 运行错误: {e}")

finally:
    try:
        depth_stream.stop()
        openni2.unload()
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass