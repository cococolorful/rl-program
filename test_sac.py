import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from camera import update_camera_view
from assit import assit
from matplotlib.path import Path
from env import CameraControlEnv
from SAC import SAC  # 导入SAC模型

matplotlib.use("TKAgg")

# 创建飞机形状的Path数据
path_data = [
    (Path.MOVETO, [0.0, -0.3]),
    (Path.LINETO, [0.3, 0.0]),
    (Path.LINETO, [-0.3, 0.0]),
    (Path.LINETO, [0.0, -0.3]),
]

codes, verts = zip(*path_data)
path = Path(verts, codes)

# 检查是否可用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建相机控制环境
env = CameraControlEnv()

# SAC算法的参数
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

sac_args = {
    'gamma': 0.999,  # 折扣率，控制未来奖励的重要性0.99
    'tau': 0.005,  # 软更新参数，控制目标网络的平滑更新
    'alpha': 0.4,  # 熵调节参数，控制策略的探索程度0.2
    'policy': 'Gaussian',  # 策略类型，可以是Gaussian或Deterministic策略
    'target_update_interval': 1,  # 目标网络更新的频率
    'automatic_entropy_tuning': True,  # 是否自动调节熵参数
    'cuda': True,
    'hidden_size': 1024,  # Critic和Actor网络的隐藏层大小
    'lr': 0.001,  # 学习率0.0001
    'buffer_capacity': 10000000,  # 添加buffer_capacity属性
    'seed': 256  # 添加seed属性
}

def store_state(state, n):
    all_states.append(state)
    n[0] += 1

# 在这里初始化SAC代理
agent = SAC(state_dim, env.action_space, sac_args)

# 定义目录路径以加载模型参数
load_dir = 'sac_model'
load_episode = 1000  # 你想加载的训练轮次对应的模型参数

# 加载模型参数
agent.load_model(load_dir, 'sac', load_episode, evaluate=True)

# 创建绘图对象和散点对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 初始化散点对象
scatter = None

# 在代码开始前，添加一个空的文本注释对象
annotation1 = ax.text2D(1.00, 0.87, '', transform=ax.transAxes, fontsize=14, color='black')
annotation2 = ax.text2D(-0.20, 0.75, '', transform=ax.transAxes, fontsize=14, color='black')
annotation3 = ax.text2D(0.35, 0.90, '', transform=ax.transAxes, fontsize=14, color='black')
annotation4 = ax.text2D(1.00, 0.80, '', transform=ax.transAxes, fontsize=16, color='red', weight='bold')

# 初始化all_states列表
all_states = []
actions = []

# 测试代理
num_episodes = 1

# 在代码开始前，添加一个变量来跟踪是否要切换到手动动作
fone = True

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    n = [0]  # 将 n 包装在一个列表中，以便在函数内部可以更新它

    while not done:
        observation = state[:5]
        azimuth, elevation, focal_length, gud_a, gud_e, v, a, x, y, z, current_t = state
        azimuth = round(azimuth, 2)
        elevation = round(elevation, 2)
        camera_pos = np.array([0, 0, 0])

        # 计算目标相对于相机的位置矢量
        target_vector = np.array([x, y, z]) - camera_pos

        # 计算俯角和仰角
        distance = np.linalg.norm(target_vector)
        elevation_rad = np.arcsin(target_vector[2] / distance)
        azimuth_rad = np.arctan2(target_vector[1], target_vector[0])

        elevation_deg = np.rad2deg(elevation_rad)
        elevation_deg = round(elevation_deg, 2)

        azimuth_deg = np.rad2deg(azimuth_rad)
        azimuth_deg = round(azimuth_deg, 2)

        azimuth1 = azimuth_deg - azimuth
        elevation1 = elevation_deg - elevation
        
        if fone:
            action = agent.select_action(observation, evaluate=True)  # 使用训练后的策略进行评估
        else:
            action = [azimuth1, elevation1, 0.129]  # 手动定义动作
        
        actions.append(action)

        next_state, reward, done, _ = env.step(action)
        store_state(next_state, n)
        total_reward += reward
        state = next_state

        if reward == 100:  # 检查是否满足条件
            #print(f"GOOD")
            fone = False  # 切换到手动动作

    print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")

    # 获取总帧数
    total_t = len(all_states)

# print(f"{all_states}")

# 定义绘图函数
def update_plot(frame):
    global scatter  # 声明为 global，以便在整个函数范围内都可见

    state = all_states[frame]
    azimuth, elevation, focal_length, gud_a, gud_e, v, a, x, y, z, current_t = state
    azimuth = round(azimuth, 2)
    elevation = round(elevation, 2)
    action =actions[frame]
    speed_a = action[0] * 2
    speed_a = round(speed_a, 2)
    speed_e = action[1] * 2
    speed_e = round(speed_e, 2)
    R1, R2, R3 = assit(focal_length)

    camera_pos = np.array([0, 0, 0])
    d = 0.00818
    aspect_ratio = 16/9

    # 计算目标相对于相机的位置矢量
    target_vector = np.array([x, y, z]) - camera_pos

    # 计算俯角和仰角
    distance = np.linalg.norm(target_vector)
    elevation_rad = np.arcsin(target_vector[2] / distance)
    azimuth_rad = np.arctan2(target_vector[1], target_vector[0])

    elevation_deg = np.rad2deg(elevation_rad)
    elevation_deg = round(elevation_deg, 2)

    azimuth_deg = np.rad2deg(azimuth_rad)
    azimuth_deg = round(azimuth_deg, 2)

    # 动态设置坐标系的范围
    x_margin = 100  # X轴的边界范围
    y_margin = 100  # Y轴的边界范围
    x_range = abs(x) + x_margin  # 使用绝对值来设置X轴范围
    y_range = abs(y) + y_margin  # 使用绝对值来设置Y轴范围
    ax.set_xlim(camera_pos[0] - x_range, camera_pos[0] + x_range)
    ax.set_ylim(camera_pos[1] - y_range, camera_pos[1] + y_range)
    ax.set_zlim(0, 250)

    # 添加指引线的终点坐标
    line_x = 1000 * np.cos(np.deg2rad(gud_a))
    line_y = 1000 * np.sin(np.deg2rad(gud_a))
    line_z = 1000 * np.tan(np.deg2rad(gud_e))

    # 清除前一帧的指引线
    if hasattr(ax, "guide_line"):
        for line in ax.guide_line:
            line.remove()

    # 添加新的指引线
    ax.guide_line = ax.plot([0, line_x], [0, line_y], [0, line_z], 'r', alpha=0.3)

    # 清除当前设置的内容
    annotation1.set_text("")
    distance1 = round(distance, 2)

    # 更新文本注释内容
    info_text1 = f'UAV:\n\n'\
            f'True_azimuth: {azimuth_deg} °\n\n' \
            f'True_elevation: {elevation_deg} °\n\n'\
            f'Position: {[x, y, z]}\n\n'\
            f'Height: {z} m\n\n'\
            f'Distance: {distance1} m\n\n'\
            f'v: {v} m/s  ' 

    progress1 = abs(gud_a - azimuth)
    progress1 = round(progress1, 2)
    progress2 = abs(gud_e - elevation)
    progress2 = round(progress2, 2)

    fov_adjusted1 = np.rad2deg(2 * np.arctan(0.5 * d / focal_length))
    fov_adjusted1 = round(fov_adjusted1, 2)
    fov_adjusted2 = fov_adjusted1 / aspect_ratio
    fov_adjusted2 = round(fov_adjusted2, 2)

    a1 = azimuth - fov_adjusted1 / 2
    a2 = azimuth + fov_adjusted1 / 2
    e1 = elevation - fov_adjusted2 / 2
    e2 = elevation + fov_adjusted2 / 2

    p1 = 1 if a1 <= azimuth_deg <= a2 else 0
    p2 = 1 if e1 <= elevation_deg <= e2 else 0
    oi = 1 if p1 == 1 and p2 == 1 else 0

    # 更新文本注释内容
    if distance <= R1 and oi == 1:
        text = "indentification"

    # 在R1范围内的处理代码
    elif R1 <= distance <= R2 and oi == 1:
        text = "recognition"

    # 在R2范围内的处理代码
    elif R2 <= distance <= R3 and oi == 1:
        text = "detection"

    # 在R3范围内的处理代码
    else:
        text = "undetection"

    focal_length1 = focal_length * 1000
    focal_length1 = round(focal_length1, 2)

    info_text2 = f'Guidance:\n\n'\
            f'Azimuth: {gud_a} °  Elevation: {gud_e} °\n\n' \
            f'Progress: Azi: {progress1} °  Ele:{progress2} °\n\n\n' 
    
    info_text3 = f'OE Syetem:\n\n'\
            f'Focal_length: {focal_length1} mm\n\n' \
            f'Fov_azi: {fov_adjusted1} °  Fov_ele: {fov_adjusted2}°\n\n'\
            f'Speed_azi: {speed_a} °/s  Speed_ele: {speed_e}°/s\n\n'\
            f'Azimuth: {azimuth} °  Elevation: {elevation}°'
    info_text4 = f'State: {text}'
    annotation2.set_text(info_text1)  # 设置文本内容
    annotation3.set_text(info_text2)  # 设置文本内容   
    annotation1.set_text(info_text3)
    annotation4.set_text(info_text4)

    update_camera_view(ax, d, aspect_ratio, camera_pos, azimuth, elevation, focal_length, R1, R2, R3, distance, oi)

    # if scatter is not None:
        
    # 清除前一帧的点
        # scatter.remove()

    # 在散点图中使用自定义标记并设置标记大小
    scatter = ax.scatter(x, y, z, c='black', marker=path, s = 6 * 6)

# 定义生成帧的回调函数
def animate(frame):
    update_plot(frame)

# 创建动画
anim = FuncAnimation(fig, animate, frames=total_t, interval=120, repeat=False)

plt.show()

# 关闭当前图形
plt.close(fig)