# -*- coding: utf-8 -*-
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import os
import sys

# ==========================================
# 1. 环境与路径配置 (解决 DLL 找不到的问题)
# ==========================================
# 获取当前 main.py 文件所在的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出 OpenNI2.dll 的完整路径
dll_path = os.path.join(script_dir, "OpenNI2.dll")
print(f"正在尝试加载 OpenNI，路径: {script_dir}")

# 检查文件是否存在
if not os.path.exists(dll_path):
    print("【严重错误】找不到 OpenNI2.dll！")
    print(f"请确保 OpenNI2.dll 和 OpenNI2文件夹 都在这个目录下: {script_dir}")
    sys.exit()

# 初始化 OpenNI
try:
    openni2.initialize(script_dir)
except Exception as e:
    print(f"【初始化失败】: {e}")
    sys.exit()

# ==========================================
# 2. 启动 Astra Pro 深度相机 (OpenNI)
# ==========================================
try:
    dev = openni2.Device.open_any()
    print(f"✅ 深度设备已连接: {dev.get_device_info().name}")
except Exception as e:
    print(f"【无法打开深度设备】: {e}")
    sys.exit()

# 创建深度流
depth_stream = dev.create_depth_stream()
depth_stream.start()

# 设置深度流模式
# 修复之前的 AttributeError: 使用 100 代表 1mm 深度格式
# 分辨率 640x480, 30fps
try:
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=100, resolutionX=640, resolutionY=480, fps=30))
except Exception as e:
    print(f"设置视频模式失败: {e}")

# 开启【深度-彩色】硬件对齐 (关键！)
if dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    print("✅ 深度-彩色 对齐已开启")
else:
    print("⚠️ 警告：设备不支持硬件对齐")

# ==========================================
# 3. 启动 RGB 相机 (OpenCV)
# ==========================================
# 使用你确认的索引 1，并加上 cv2.CAP_DSHOW 提高兼容性
print("正在尝试打开 RGB 相机 (Index 1)...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 设置分辨率以匹配深度图 (Astra Pro 最佳匹配是 640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ 无法打开 RGB 摄像头 (索引 1)")
    # 紧急备用方案
    print("尝试备用索引 0 ...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 所有尝试均失败，程序退出")
        sys.exit()

print("✅ RGB 相机已启动")

def get_depth_frame():
    """读取并转换深度帧为 numpy 数组"""
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)
    img.shape = (480, 640) # 必须与 set_video_mode 一致
    return img

print("\n=== 程序运行中 (按 'q' 退出) ===")
print("提示：将物体放在画面中心十字处，查看控制台输出的 XYZ 坐标")

# ==========================================
# 4. 主循环
# ==========================================
try:
    while True:
        # --- 读取 RGB ---
        ret, color_img = cap.read()
        if not ret:
            continue

        # --- 读取 Depth ---
        # 如果没有深度数据，可能会阻塞在这里，直到有数据为止
        depth_img = get_depth_frame()

        # --- 核心逻辑：获取中心点坐标 ---
        h, w = color_img.shape[:2]
        center_x, center_y = w // 2, h // 2 # 画面中心点 (u, v)

        # 获取该像素点的深度值 (单位: 毫米)
        depth_value = depth_img[center_y, center_x]
        
        # 坐标转换：像素坐标 (u,v,d) -> 世界坐标 (x,y,z)
        if depth_value > 0:
            x, y, z = openni2.convert_depth_to_world(depth_stream, center_x, center_y, depth_value)
            # 格式化显示的文本
            info_text = f"XYZ: ({x:.0f}, {y:.0f}, {z:.0f}) mm"
            color_text = (0, 255, 0) # 绿色代表有效
        else:
            info_text = "XYZ: Too Close/Far" # 深度无效（太近或太远，或者黑色物体吸光）
            color_text = (0, 0, 255) # 红色代表无效

        # --- 可视化 ---
        # 1. 处理深度图用于显示 (归一化到 0-255 并上色)
        depth_show = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_show = cv2.applyColorMap(depth_show, cv2.COLORMAP_JET)

        # 2. 在 RGB 图上画十字和坐标
        cv2.drawMarker(color_img, (center_x, center_y), color_text, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(color_img, info_text, (center_x + 10, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)

        # 3. 显示窗口
        cv2.imshow("RGB Camera", color_img)
        cv2.imshow("Depth Map", depth_show)

        # 按 'q' 退出
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # 资源释放
    print("正在关闭设备...")
    depth_stream.stop()
    openni2.unload()
    cap.release()
    cv2.destroyAllWindows()
    print("程序已结束")