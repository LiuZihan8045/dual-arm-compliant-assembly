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
# 1. 用户配置区
# ==========================================
# 镜像开关 (根据你之前的调试结果修改)
NEED_MIRROR_DEPTH = False  

# Aruco 边长 (米) - 请务必量准
MARKER_SIZE = 0.048 

# 相机内参
fx, fy = 616.7092, 618.3210
cx, cy_opt = 328.3358, 249.1239
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy_opt], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([[-0.04093593170768245, 1.7737048617046605, 0.0030425822102364388, -0.002016729524233336, -6.27798179559079]])

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

def rotation_matrix_to_euler_angles(R):
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
    if dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    try:
        depth_stream.set_mirroring_enabled(False)
    except: pass
    print("✅ 深度引擎就绪")
except Exception as e:
    print(f"❌ OpenNI 错误: {e}")
    sys.exit()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# 状态变量
last_snapshot_time = 0
snapshot_feedback_duration = 1.0 # 提示文字显示秒数

# ==========================================
# 4. 主循环
# ==========================================
print("\n=============================================")
print("  操作说明:")
print("  [q] 退出程序")
print("  [g] 获取当前 6D 姿态和深度信息 (打印到控制台)")
print("=============================================\n")

try:
    while True:
        # 1. 读取 RGB
        ret, frame_rgb = cap.read()
        if not ret: continue
        
        # 初始化当帧数据容器
        current_data_list = [] # 用于存储当前帧检测到的所有物体信息

        # 2. 读取深度
        try:
            d_frame = depth_stream.read_frame()
            if d_frame.dataSize != 614400: continue
            d_frame_data = d_frame.get_buffer_as_uint16()
            d_img_raw = np.frombuffer(d_frame_data, dtype=np.uint16).reshape(480, 640)
            if NEED_MIRROR_DEPTH:
                d_img_raw = cv2.flip(d_img_raw, 1)
            depth_valid = True
        except:
            depth_valid = False
            d_img_raw = np.zeros((480, 640), dtype=np.uint16)

        # 3. 深度热力图
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
                    # A. 提取基础数据
                    rvec, tvec = rvecs[i][0], tvecs[i][0]
                    marker_id = ids[i][0]
                    
                    c = corners[i][0]
                    cx, cy = int((c[0][0] + c[2][0]) / 2), int((c[0][1] + c[2][1]) / 2)
                    
                    # B. 计算 6D 姿态 (Translation + Rotation)
                    # 位置 (XYZ) - 单位: 毫米
                    pos_x = tvec[0] * 1000
                    pos_y = tvec[1] * 1000
                    pos_z = tvec[2] * 1000
                    
                    # 姿态 (RPY) - 单位: 度
                    R_mat, _ = cv2.Rodrigues(rvec)
                    roll, pitch, yaw = rotation_matrix_to_euler_angles(R_mat)
                    
                    # C. 深度相机测量
                    depth_sensor_val = get_robust_depth(d_img_raw, cx, cy)
                    
                    # D. 保存数据到临时列表 (供按 'g' 时使用)
                    obj_info = {
                        "id": marker_id,
                        "pos": (pos_x, pos_y, pos_z),
                        "rpy": (roll, pitch, yaw),
                        "depth_sensor": depth_sensor_val
                    }
                    current_data_list.append(obj_info)
                    
                    # E. 实时绘图
                    cv2.drawFrameAxes(frame_rgb, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    aruco.drawDetectedMarkers(frame_rgb, corners)
                    cv2.circle(frame_rgb, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # 显示文字
                    info_text = f"ID:{marker_id} Z:{pos_z:.0f}mm"
                    cv2.putText(frame_rgb, info_text, (cx-40, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 右图红点
                    cv2.circle(depth_vis_color, (cx, cy), 5, (0, 0, 255), -1)

        # 5. 显示反馈文字 (如果刚按过 g)
        if time.time() - last_snapshot_time < snapshot_feedback_duration:
            cv2.putText(frame_rgb, "SNAPSHOT SAVED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 6. 拼接显示
        combined = np.hstack((frame_rgb, depth_vis_color))
        cv2.imshow("Press 'g' to Capture", combined)

        # 7. 按键处理
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        
        elif key == ord('g'):
            # === 触发数据输出 ===
            last_snapshot_time = time.time()
            print("\n" + "="*40)
            print(f"📸 捕获时间: {time.strftime('%H:%M:%S', time.localtime())}")
            
            if len(current_data_list) > 0:
                for item in current_data_list:
                    print(f"🔸 Marker ID: {item['id']}")
                    print(f"   [6D Position] X: {item['pos'][0]:.1f}, Y: {item['pos'][1]:.1f}, Z: {item['pos'][2]:.1f} (mm)")
                    print(f"   [6D Rotation] Roll: {item['rpy'][0]:.1f}, Pitch: {item['rpy'][1]:.1f}, Yaw: {item['rpy'][2]:.1f} (deg)")
                    print(f"   [DepthSensor] Distance: {item['depth_sensor']} mm")
                    print("-" * 30)
            else:
                print("⚠️ 当前画面未检测到任何 Aruco 码")
            print("="*40 + "\n")

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