import cv2
import sys
from ultralytics import YOLO

print("--- 正在检查 YOLO 环境 ---")

# 1. 尝试导入核心库
try:
    import torch
    import numpy
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ NumPy 版本: {numpy.__version__}")
    # 检查 CUDA 是否可用 (可选)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 运行设备: {device}")
except ImportError as e:
    print(f"❌ 环境损坏，缺少库: {e}")
    sys.exit()
except Exception as e:
    print(f"❌ 环境异常 (可能是 NumPy 版本问题): {e}")
    print("💡 建议尝试运行: pip install \"numpy<2\"")
    sys.exit()

# 2. 加载模型
print("⏳ 正在加载 YOLOv8n 模型...")
try:
    model = YOLO('yolov8n.pt')
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit()

# 3. 打开自带摄像头 (Index 0)
print("📷 正在打开笔记本摄像头...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Windows 建议加上 CAP_DSHOW

if not cap.isOpened():
    print("❌ 无法打开摄像头 (Index 0)")
    sys.exit()

print("\n=== 开始测试 (按 'q' 退出) ===")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取画面")
        break

    # YOLO 推理
    # verbose=False 让控制台清净一点
    results = model(frame, verbose=False)

    # 绘制结果 (ultralytics 自带的绘图功能，最简单)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Environment Test", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ 测试结束，环境正常。")