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
import ctypes

# 防止库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. 用户配置
# ==========================================
MARKER_SIZE = 0.05 
fx, fy = 580.0, 580.0
cx, cy_opt = 320.0, 240.0
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy_opt], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# ==========================================
# 2. 辅助函数
# ==========================================
def get_robust_depth(depth_frame, x, y, roi_size=2):
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
# 3. 初始化
# ==========================================
print("--- 正在启动系统 (防闪退版) ---")

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
    if dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    print("✅ 深度引擎就绪")
except Exception as e:
    print(f"❌ OpenNI 错误: {e}")
    sys.exit()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# ==========================================
# 4. 主循环 (加了异常捕获)
# ==========================================
print("\n=== 系统运行中 (按 'q' 退出) ===")

try:
    while True:
        # --- 安全读取 RGB ---
        ret, frame_rgb = cap.read()
        if not ret: continue

        # --- 安全读取 Depth (关键修改点) ---
        try:
            d_frame = depth_stream.read_frame()
            
            # 【防闪退核心】检查 buffer 大小是否正确
            # 640 * 480 * 2 bytes (16bit) = 614400 bytes
            if d_frame.dataSize != 614400:
                print(f"⚠️ 跳过坏帧 (Size: {d_frame.dataSize})")
                continue
                
            d_frame_data = d_frame.get_buffer_as_uint16()
            d_img_raw = np.frombuffer(d_frame_data, dtype=np.uint16).reshape(480, 640)
            
            # 只有拿到了正确的数据，才做后续处理，否则用全黑图代替
            depth_valid = True
        except Exception as e:
            # 如果深度读取出错，打印错误但不退出
            print(f"⚠️ 深度读取警告: {e}")
            depth_valid = False
            # 创建一个全黑的深度图防止绘图报错
            d_img_raw = np.zeros((480, 640), dtype=np.uint16)

        # --- 可视化处理 ---
        if depth_valid:
            depth_vis = cv2.normalize(d_img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        else:
            depth_vis_color = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(depth_vis_color, "Depth Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- Aruco 处理 ---
        if depth_valid: # 只有深度有效才去算距离，防止不同步
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                for i in range(len(ids)):
                    rvec, tvec = rvecs[i][0], tvecs[i][0]
                    
                    # 获取中心点
                    c = corners[i][0]
                    cx, cy = int((c[0][0] + c[2][0]) / 2), int((c[0][1] + c[2][1]) / 2)
                    
                    # 深度融合
                    depth_z = get_robust_depth(d_img_raw, cx, cy)
                    
                    # 绘图
                    cv2.drawFrameAxes(frame_rgb, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    aruco.drawDetectedMarkers(frame_rgb, corners)
                    cv2.putText(frame_rgb, f"Z: {depth_z}mm", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 在右侧深度图上也画个圈
                    cv2.circle(depth_vis_color, (cx, cy), 5, (255, 255, 255), 2)

        # --- 拼接显示 ---
        combined_view = np.hstack((frame_rgb, depth_vis_color))
        cv2.imshow("Astra Pro Safe Mode", combined_view)

        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"❌ 主循环发生不可预知的错误: {e}")

finally:
    print("正在关闭...")
    try:
        depth_stream.stop()
        openni2.unload()
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass