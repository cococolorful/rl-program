from SAC import SAC  # SAC类是你提供的SAC算法实现
from env import CameraControlEnv  # CameraControlEnv是你的自定义环境
import numpy as np
import json
import os

# 创建相机控制环境
env = CameraControlEnv()

# SAC算法的参数
state_dim = env.observation_space.shape[0]  # 观察空间为5维
action_dim = env.action_space.shape[0]
max_action = env.action_space.high
min_action = env.action_space.low  # 添加min_action参数

sac_args = {
    'gamma': 0.99,  # 折扣率，控制未来奖励的重要性0.99
    'tau': 0.005,  # 软更新参数，控制目标网络的平滑更新
    'alpha': 0.4,  # 熵调节参数，控制策略的探索程度0.2
    'policy': 'Deterministic',  # 策略类型，可以是Gaussian或Deterministic策略
    'target_update_interval': 5,  # 目标网络更新的频率
    'automatic_entropy_tuning': True,  # 是否自动调节熵参数
    'cuda': True,
    'hidden_size': 1024,  # Critic和Actor网络的隐藏层大小
    'lr': 0.001,  # 学习率0.0001
    'buffer_capacity': 10000000,  # 添加buffer_capacity属性
    'seed': 256  # 添加seed属性
}

# 在这里初始化SAC代理
agent = SAC(state_dim, env.action_space, sac_args)

buffer_save_path = 'checkpoints/sac_buffer_camera_control_.pkl'

# 尝试加载 Replay Buffer
if os.path.exists(buffer_save_path):
    print(f"Loading Replay Buffer from {buffer_save_path}...")
    agent.buffer.load_buffer(buffer_save_path)
else:
    print("No saved Replay Buffer found. Starting with an empty buffer.")

# 构建文件的绝对路径
file_path = 'tmpkw.json'

# 读取 JSON 文件
with open(file_path, "r") as file:
    data = json.load(file)

# 打印输出 Defender 的数据
defender_data = data.get("Defender", [])
defender_var_count = len(defender_data)
#print(defender_data)

# 循环与代理交互并手动输入经验
observation = env.reset()
total_reward = 0
done = False
n = 0

while not done:

    if n < defender_var_count - 1:
        action = [defender_data[n + 1][i] - defender_data[n][i] for i in range(len(defender_data[0]))]
        n = n + 1
        #print(n)
    else:
        action = [0, 0, 0]
    #print(action)

    next_observation, reward, done, _ = env.step(action)
    # 打印获得的奖励
    # print(f"Reward: {reward}")
    agent.buffer.push(observation, action, reward, next_observation, done)
    # print(f"{len(agent.buffer)}")
    # 保存 Replay Buffer
    total_reward += reward
    
print(f"Total Reward: {total_reward}")

# 关闭环境
env.close()

agent.buffer.save_buffer('camera_control')