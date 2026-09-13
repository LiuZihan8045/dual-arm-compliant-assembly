# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

# =================配置区=================
# 如果中文显示乱码，请修改这里的字体 (Windows通常是 SimHei 或 Microsoft YaHei)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 流程步骤定义 (严格按照你的要求)
steps = [
    "初始化\n(System Init)", 
    "获取 RGB 帧\n(Get RGB Frame)", 
    "获取深度帧\n(Get Depth Frame)", 
    "ArUco 检测\n(ArUco Detection)", 
    "姿态解算 (PnP)\n(Pose Estimation)", 
    "深度提取\n(Depth Extraction)", 
    "融合校验\n(Fusion Verify)", 
    "渲染显示\n(Render Display)"
]

def draw_flowchart():
    # 创建画布
    fig, ax = plt.subplots(figsize=(6, 12)) # 长宽比例适合垂直流程图
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.5 + 1)
    ax.axis('off') # 关闭坐标轴

    # 绘制参数
    box_w = 6   # 框宽
    box_h = 0.8 # 框高
    center_x = 5
    start_y = len(steps) * 1.5 - 0.5
    gap = 1.5

    # 循环绘制每一个节点
    for i, text in enumerate(steps):
        y = start_y - i * gap
        
        # 1. 绘制方框
        # 圆角矩形 (FancyBboxPatch) 看起来更像论文图表
        box = patches.FancyBboxPatch(
            (center_x - box_w/2, y - box_h/2), 
            box_w, box_h,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            ec="black", fc="white", lw=1.5
        )
        ax.add_patch(box)
        
        # 2. 绘制文字
        ax.text(center_x, y, text, ha='center', va='center', fontsize=12, fontweight='bold')

        # 3. 绘制向下箭头 (除了最后一个)
        if i < len(steps) - 1:
            arrow_start_y = y - box_h/2 - 0.1
            arrow_end_y = (start_y - (i+1) * gap) + box_h/2 + 0.1
            ax.arrow(
                center_x, arrow_start_y, 
                0, arrow_end_y - arrow_start_y, 
                head_width=0.2, head_length=0.2, fc='black', ec='black', lw=1.5,
                length_includes_head=True
            )

    # 4. 绘制循环箭头 (从"渲染显示"回到"获取RGB帧")
    # 计算起点和终点坐标
    last_y = start_y - (len(steps)-1) * gap
    target_y = start_y - 1 * gap # Index 1 is RGB Frame
    
    # 画折线箭头
    line_x = center_x + box_w/2 + 0.5 # 右侧绕行
    
    # 路径: 底部右出 -> 向上 -> 目标右入
    # 底部横线
    ax.plot([center_x + box_w/2, line_x], [last_y, last_y], 'k-', lw=1.5)
    # 垂直竖线
    ax.plot([line_x, line_x], [last_y, target_y], 'k-', lw=1.5)
    # 顶部箭头 (指向 RGB Frame)
    ax.arrow(
        line_x, target_y, 
        (center_x + box_w/2) - line_x + 0.2, 0, # +0.2 为了留出箭头空间
        head_width=0.2, head_length=0.2, fc='black', ec='black', lw=1.5,
        length_includes_head=True
    )
    
    # 添加"Loop"文字
    ax.text(line_x + 0.2, (last_y + target_y)/2, "Next Frame", ha='left', va='center', fontsize=10, style='italic')

    # 保存图片
    plt.tight_layout()
    save_path = 'system_flowchart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 流程图已生成并保存为: {save_path}")
    plt.show()

if __name__ == "__main__":
    draw_flowchart()