# -*- coding: utf-8 -*-
import os
import sys

# 【修复 1】解决 PyTorch 和 OpenCV 的 OpenMP 冲突 (防止静默崩溃的关键!)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
from ultralytics import YOLO
import time

# 辅助函数：强制刷新打印，确保你能看到每一步
def log(msg):
    print(msg, flush=True)

def get_robust_depth(depth_frame, x, y, roi_size=4):
    h, w = depth_frame.shape
    x_min = max(0, x - roi_size)
    x_max = min(w, x + roi_size)
    y_min = max(0, y - roi_size)
    y_max = min(h, y + roi_size)
    roi = depth_frame[y_min:y_max, x_min:x_max]
    valid_depths = roi[roi > 0]
    if len(valid_depths) == 0: return 0
    return np.median(valid_depths)

log("--- 程序启动 ---")

# ==========================================
# 1. 先启动 RGB 相机 (调整顺序)
# ==========================================
log("1. 正在初始化 RGB 相机...")
# 尝试索引 1
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    log("⚠️ 索引 1 失败，尝试索引 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
if not cap.isOpened():
    log("❌ 严重错误：无法打开任何摄像头")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
log(f"✅ RGB 相机已打开 (后端: {cap.getBackendName()})")

# 【修复 2】休息一下，防止 USB 拥堵
time.sleep(1.0) 

# ==========================================
# 2. 再启动 OpenNI 深度相机
# ==========================================
log("2. 正在初始化深度相机 (OpenNI)...")
script_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(script_dir, "OpenNI2.dll")

if not os.path.exists(dll_path):
    log("❌ 找不到 OpenNI2.dll")
    sys.exit()

try:
    openni2.initialize(script_dir)
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=100, resolutionX=640, resolutionY=480, fps=30))
    
    if dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        log("✅ 深度-彩色对齐已开启")
    
    log(f"✅ 深度相机就绪: {dev.get_device_info().name}")

except Exception as e:
    log(f"❌ OpenNI 初始化失败: {e}")
    sys.exit()

# ==========================================
# 3. 最后加载模型
# ==========================================
log("3. 正在加载 YOLO 模型...")
try:
    model = YOLO('yolov8n.pt')
    log("✅ 模型加载成功")
except Exception as e:
    log(f"❌ 模型加载失败: {e}")
    sys.exit()

# ==========================================
# 4. 主循环
# ==========================================
log("\n=== 开始运行 (按 'q' 退出) ===")
fps_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            log("⚠️ RGB 读取失败")
            continue
        
        # 读取深度
        d_frame = depth_stream.read_frame()
        d_frame_data = d_frame.get_buffer_as_uint16()
        d_img = np.frombuffer(d_frame_data, dtype=np.uint16).reshape(480, 640)

        # YOLO 推理
        results = model(frame, stream=True, verbose=False, conf=0.5)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # 获取真实坐标
                dist_mm = get_robust_depth(d_img, cx, cy)
                
                if dist_mm > 0:
                    rw_x, rw_y, rw_z = openni2.convert_depth_to_world(depth_stream, cx, cy, dist_mm)
                    coord_text = f"X:{rw_x:.0f} Y:{rw_y:.0f} Z:{rw_z:.0f}"
                    color = (0, 255, 0)
                else:
                    coord_text = "N/A"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1-20), (x1+200, y1), color, -1)
                cv2.putText(frame, f"{label} {coord_text}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 显示 FPS
        fps_count += 1
        if time.time() - start_time > 1.0:
            print(f"FPS: {fps_count}", flush=True)
            fps_count = 0
            start_time = time.time()

        cv2.imshow("YOLO + OpenNI", frame)

        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    log(f"❌ 运行时错误: {e}")

finally:
    log("正在退出...")
    depth_stream.stop()
    openni2.unload()
    cap.release()
    cv2.destroyAllWindows()