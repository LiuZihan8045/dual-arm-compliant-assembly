# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import glob
import sys

# ==========================================
# 1. 配置区 (请根据你的棋盘格修改)
# ==========================================
# 棋盘格内角点数量 (列, 行) -> 数黑白交界的十字点
CHESSBOARD_SIZE = (8, 5) 

# 每个格子的实际边长 (单位: 毫米)
# 虽然这一步只影响平移向量，不影响内参 fx/fy，但填准了好习惯
SQUARE_SIZE = 26  

# 图片保存路径
SAVE_DIR = "calibration_imgs"

# ==========================================
# 2. 准备工作
# ==========================================
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 设置寻找亚像素角点的参数，采用停止准则 (最大循环次数30, 最大误差0.001)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备物体点 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
# 将世界坐标系建在标定板上，Z=0
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = [] # 真实世界中的 3D 点
imgpoints = [] # 图像平面上的 2D 点

# ==========================================
# 3. 第一阶段：采集图片
# ==========================================
print(f"--- 相机标定工具 ---")
print(f"棋盘格设定: {CHESSBOARD_SIZE}")
print("操作指南:")
print("  [c] 拍照 (请拍摄 15-20 张不同角度)")
print("  [q] 结束拍照并开始计算")

# 打开 RGB 相机
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): sys.exit("❌ 无法打开相机")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret: continue

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 实时检测角点 (为了帮你对焦)
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # 如果检测到了，画出来给你看
    if ret_corners:
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)
        cv2.putText(display_frame, "Ready to Capture!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Searching Board...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(display_frame, f"Images: {img_count}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Camera Calibration (Press 'c' to capture)", display_frame)

    key = cv2.waitKey(1)
    
    # 按 'c' 拍照
    if key == ord('c'):
        if ret_corners:
            filename = os.path.join(SAVE_DIR, f"img_{img_count:02d}.jpg")
            cv2.imwrite(filename, frame) # 保存原始未画线的图
            print(f"📸 已保存: {filename}")
            img_count += 1
            # 闪烁一下提示保存成功
            cv2.rectangle(display_frame, (0,0), (640,480), (255,255,255), 10)
            cv2.imshow("Camera Calibration (Press 'c' to capture)", display_frame)
            cv2.waitKey(100)
        else:
            print("⚠️ 未检测到棋盘格，无法拍照！请调整角度。")

    # 按 'q' 退出
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if img_count < 5:
    print("❌ 图片太少，无法标定 (至少需要 5 张)")
    sys.exit()

# ==========================================
# 4. 第二阶段：计算参数
# ==========================================
print(f"\n⏳ 正在计算标定参数 (使用 {img_count} 张图片)...")

images = glob.glob(os.path.join(SAVE_DIR, '*.jpg'))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找角点
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret == True:
        objpoints.append(objp)
        # 亚像素级精确化 (提高精度)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
    else:
        print(f"⚠️ 警告: 图片 {fname} 未能识别角点，已跳过")

# 核心标定函数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# ==========================================
# 5. 输出结果
# ==========================================
print("\n" + "="*50)
print("✅ 标定完成！请复制以下参数到你的主程序中:")
print("="*50)

print(f"\n# 替换 camera_matrix:")
print("fx = {:.4f}".format(mtx[0,0]))
print("fy = {:.4f}".format(mtx[1,1]))
print("cx = {:.4f}".format(mtx[0,2]))
print("cy = {:.4f}".format(mtx[1,2]))

print(f"\n# 替换 dist_coeffs (畸变系数):")
print(f"dist_coeffs = np.array({dist.tolist()})")

print(f"\n# 标定误差 (越小越好，通常 < 1.0): {ret:.4f}")
print("="*50)