import cv2

print("正在检测可用摄像头 (按 'q' 退出当前画面测试下一个)...")

# 我们测试从索引 0 到 3
for index in range(4):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) # 加上 CAP_DSHOW 对 Windows 更友好
    if not cap.isOpened():
        print(f"❌ 摄像头索引 {index}: 无法打开")
        continue
    
    # 获取一下分辨率，确认它活着
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"✅ 摄像头索引 {index}: 成功打开 (分辨率 {int(w)}x{int(h)}) -> 请看弹出的画面")
    
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera Index {index}", frame)
            # 在画面上写上编号
            cv2.putText(frame, f"Index: {index}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

print("检测结束")