import os
import sys
import time

print("⚠️ 警告：这将强制关闭所有正在运行的 Python 程序！")
print("⚠️ 请确保你没有正在运行的其他重要代码。")
print("---")
print("正在尝试清理僵尸进程并释放摄像头...")

# Windows 专用命令
# /F = 强制终止
# /IM = 指定镜像名称 (python.exe)
# /T = 终止子进程
cmd = "taskkill /F /IM python.exe /T"

try:
    # 执行杀进程命令
    os.system(cmd)
except Exception as e:
    print(f"清理出错: {e}")

# 注意：
# 因为这个脚本本身也是 python.exe，
# 所以运行到 os.system(cmd) 这一行时，
# 它不仅会杀掉别的僵尸程序，也会把自己“自杀”掉。
# 这是正常的！只要窗口消失，就说明清理成功了。