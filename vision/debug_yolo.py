# -*- coding: utf-8 -*-
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import os
import sys

# 强制刷新打印，确保你能看到死前的最后遗言
def log(msg):
    print(msg, flush=True)

log("--- 步骤 1: 开始导入 Ultralytics ---")
try:
    from ultralytics import YOLO
    log("✅ Ultralytics 导入成功")
except Exception as e:
    log(f"❌ 导入失败: {e}")
    sys.exit()

# 初始化 OpenNI
log("--- 步骤 2: 初始化 OpenNI ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(script_dir, "OpenNI2.dll")
if not os.path.exists(dll_path):
    log("❌ 找不到 OpenNI2.dll")
    sys.exit()

try:
    openni2.initialize(script_dir)
    dev = openni2.Device.open_any()
    log(f"✅ 深度相机已连接: {dev.get_device_info().name}")
    
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=100, resolutionX=640, resolutionY=480, fps=30))
    log("✅ 深度流已启动")
except Exception as e:
    log(f"❌ OpenNI 初始化失败: {e}")
    sys.exit()

# 尝试打开 RGB 相机
log("--- 步骤 3: 准备打开 RGB 相机 ---")
try:
    # 这一步最容易崩，我们加个 Try
    log("正在调用 cv2.VideoCapture(1)...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    log(f"VideoCapture 返回对象: {cap}")
    
    if not cap.isOpened():
        log("⚠️ 索引 1 打开失败，尝试索引 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            log("❌ 所有摄像头都无法打开")
            sys.exit()
    
    log(f"✅ RGB 相机打开成功 (Backend: {cap.getBackendName()})")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

except Exception as e:
    log(f"❌ RGB 相机部分发生异常: {e}")
    sys.exit()

# 加载模型
log("--- 步骤 4: 加载 YOLO 模型 ---")
try:
    model = YOLO('yolov8n.pt')
    log("✅ 模型加载完毕")
except Exception as e:
    log(f"❌ 模型加载失败 (可能是网络问题): {e}")
    sys.exit()

log("--- 步骤 5: 进入主循环 (按 'q' 退出) ---")
while True:
    ret, frame = cap.read()
    if not ret:
        log("❌ 无法读取 RGB 帧")
        break
    
    # 简单测试一下深度读取
    d_frame = depth_stream.read_frame()
    
    cv2.imshow("Debug Test", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 清理
cap.release()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
log("✅ 程序正常结束")