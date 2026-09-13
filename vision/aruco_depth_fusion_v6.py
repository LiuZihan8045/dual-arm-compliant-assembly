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
# 1. 用户配置区 (已更新你的 0.53 误差级参数)
# ==========================================
# 目标帧率 (保持稳定)
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS

# 镜像开关 (如果热力图反了就改为 True)
NEED_MIRROR_DEPTH = False  

# 【⚠️警告】Aruco 边长 (单位: 米) 
# 如果这一步填错，哪怕标定误差是 0，测出来的距离也是错的！
# 请务必用尺子量准！
MARKER_SIZE = 0.048 

# --- 你的最新标定参数 (Err: 0.5323) ---
fx = 608.9058
fy = 610.0011
cx = 316.8806
cy = 242.3447

# 内参矩阵
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)

# 畸变系数
dist_coeffs = np.array([[-0.04892722652797064, 1.5029256833676072, -0.0008574036814161698, -0.008930999672799969, -4.475786158949599]])

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
print(f"--- 启动高精度测量系统 (Err: 0.53px) ---")

# A. 启动 RGB
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): sys.exit("❌ RGB 无法启动")

cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
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

last_snapshot_time = 0
snapshot_feedback_duration = 1.0
last_frame_time = 0

# ==========================================
# 4. 主循环
# ==========================================
print("\n=== 系统就绪 (按 'g' 获取高精度数据) ===")

try:
    while True:
        # 限速
        current_time = time.time()
        if current_time - last_frame_time < FRAME_INTERVAL:
            time.sleep(0.005)
            continue
        last_frame_time = current_time

        # 1. 读取 RGB
        ret, frame_rgb = cap.read()
        if not ret: continue
        
        current_data_list = [] 

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
                    rvec, tvec = rvecs[i][0], tvecs[i][0]
                    marker_id = ids[i][0]
                    c = corners[i][0]
                    cx, cy = int((c[0][0] + c[2][0]) / 2), int((c[0][1] + c[2][1]) / 2)
                    
                    pos_x, pos_y, pos_z = tvec[0]*1000, tvec[1]*1000, tvec[2]*1000
                    R_mat, _ = cv2.Rodrigues(rvec)
                    roll, pitch, yaw = rotation_matrix_to_euler_angles(R_mat)
                    depth_sensor_val = get_robust_depth(d_img_raw, cx, cy)
                    
                    obj_info = {
                        "id": marker_id,
                        "pos": (pos_x, pos_y, pos_z),
                        "rpy": (roll, pitch, yaw),
                        "depth_sensor": depth_sensor_val
                    }
                    current_data_list.append(obj_info)
                    
                    # 绘图
                    cv2.drawFrameAxes(frame_rgb, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    aruco.drawDetectedMarkers(frame_rgb, corners)
                    cv2.circle(frame_rgb, (cx, cy), 5, (0, 0, 255), -1)
                    
                    text_z = f"Z:{pos_z:.1f}mm"
                    cv2.putText(frame_rgb, text_z, (cx-40, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 热力图对齐指示
                    cv2.circle(depth_vis_color, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(depth_vis_color, f"{depth_sensor_val}mm", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 5. 反馈提示
        if time.time() - last_snapshot_time < snapshot_feedback_duration:
            cv2.putText(frame_rgb, "CAPTURED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # 6. 显示
        combined = np.hstack((frame_rgb, depth_vis_color))
        cv2.imshow("Astra Pro High Precision", combined)

        # 7. 按键
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('g'):
            last_snapshot_time = time.time()
            print("\n" + "="*50)
            print(f"📸 测量数据 [Time: {time.strftime('%H:%M:%S')}]")
            if len(current_data_list) > 0:
                for item in current_data_list:
                    print(f"🔸 Marker ID: {item['id']}")
                    print(f"   [位置 XYZ] {item['pos'][0]:.2f}, {item['pos'][1]:.2f}, {item['pos'][2]:.2f} mm")
                    print(f"   [姿态 RPY] {item['rpy'][0]:.2f}, {item['rpy'][1]:.2f}, {item['rpy'][2]:.2f} deg")
                    print(f"   [实测深度] {item['depth_sensor']} mm")
                    
                    # 自动精度检查
                    diff = abs(item['pos'][2] - item['depth_sensor'])
                    if diff < 10:
                        print(f"   ✅ Z轴精度极佳 (偏差 {diff:.1f}mm)")
                    elif diff < 20:
                        print(f"   🆗 Z轴精度合格 (偏差 {diff:.1f}mm)")
                    else:
                        print(f"   ⚠️ Z轴偏差较大 ({diff:.1f}mm) -> 请检查 MARKER_SIZE")
                    print("-" * 30)
            else:
                print("⚠️ 未检测到 Marker")
            print("="*50 + "\n")

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