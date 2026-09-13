# -*- coding: utf-8 -*-
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
from ultralytics import YOLO
import os
import sys

# ==========================================
# 0. 辅助函数：获取鲁棒的深度值
# ==========================================
def get_robust_depth(depth_frame, x, y, roi_size=4):
    """
    在 (x,y) 周围取 roi_size x roi_size 的区域，
    去除 0 值后取中位数，防止单点噪点导致距离为 0。
    """
    height, width = depth_frame.shape
    
    # 边界检查
    if x < roi_size or x > width - roi_size or y < roi_size or y > height - roi_size:
        return 0

    # 切片取区域
    roi = depth_frame[y-roi_size : y+roi_size+1, x-roi_size : x+roi_size+1]
    
    # 提取有效深度值 (大于0)
    valid_depths = roi[roi > 0]
    
    if len(valid_depths) == 0:
        return 0 # 如果整个区域都是盲区
    
    # 返回中位数 (比平均值更抗干扰)
    return np.median(valid_depths)

# ==========================================
# 1. 初始化 OpenNI (硬件驱动)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(script_dir, "OpenNI2.dll")

if not os.path.exists(dll_path):
    print("❌ 错误：找不到 OpenNI2.dll，请检查路径。")
    sys.exit()

try:
    openni2.initialize(script_dir)
    dev = openni2.Device.open_any()
    print(f"✅ 深度相机已连接: {dev.get_device_info().name}")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    sys.exit()

# 配置深度流
depth_stream = dev.create_depth_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=100, resolutionX=640, resolutionY=480, fps=30))

# 开启硬件对齐
if dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    print("✅ 深度-彩色 对齐已开启")

# ==========================================
# 2. 初始化 RGB 相机 & YOLO
# ==========================================
# 注意：使用你刚才确认的正确索引 (通常是 1)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ 无法打开 RGB 相机")
    sys.exit()

# 加载 YOLO 模型 (第一次运行会自动下载)
print("⏳ 正在加载 YOLOv8 模型...")
model = YOLO('yolov8n.pt') 
print("✅ 模型加载完毕，开始识别！")

# ==========================================
# 3. 主循环
# ==========================================
try:
    while True:
        ret, frame = cap.read()
        if not ret: continue

        # 1. 获取深度帧
        d_frame = depth_stream.read_frame()
        d_img_raw = np.frombuffer(d_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)

        # 2. 运行 YOLO 推理
        # stream=True 稍微快一点，conf=0.5 过滤低置信度结果
        results = model(frame, stream=True, verbose=False, conf=0.5)

        # 3. 处理每个检测到的物体
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取包围框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 获取类别名称
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                # 计算中心点
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # --- 核心：获取真实空间坐标 ---
                # 使用区域采样获取深度
                dist_mm = get_robust_depth(d_img_raw, cx, cy)
                
                if dist_mm > 0:
                    # 像素 (u,v,d) -> 空间 (x,y,z)
                    real_x, real_y, real_z = openni2.convert_depth_to_world(depth_stream, cx, cy, dist_mm)
                    
                    text_coord = f"X:{real_x:.0f} Y:{real_y:.0f} Z:{real_z:.0f}"
                    color_box = (0, 255, 0) # 绿色：有效坐标
                else:
                    text_coord = "Depth: N/A"
                    color_box = (0, 0, 255) # 红色：深度无效

                # --- 绘图 ---
                # 画框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                # 画中心点
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                # 写类别名
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
                # 写坐标 (写在框的底部)
                cv2.putText(frame, text_coord, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

        # 显示画面
        cv2.imshow("Astra Pro + YOLOv8 Detection", frame)

        # 深度图可视化 (可选，用于调试)
        # d_show = cv2.normalize(d_img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # d_show = cv2.applyColorMap(d_show, cv2.COLORMAP_JET)
        # cv2.imshow("Depth", d_show)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    depth_stream.stop()
    openni2.unload()
    cap.release()
    cv2.destroyAllWindows()