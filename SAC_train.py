import torch
import os
from SAC import SAC  # SAC类是你提供的SAC算法实现
from env import CameraControlEnv  # CameraControlEnv是你的自定义环境
from env_gpu import CameraControlEnv  # CameraControlEnv是你的自定义环境

# 创建一个Tensorboard写入器
log_dir = 'logs'  # Tensorboard日志目录

# 检查是否可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1024
# 创建相机控制环境
env = CameraControlEnv(num_envs=batch_size)

# SAC算法的参数
state_dim = env.observation_space.shape[0]  # 观察空间为5维

sac_args = {
    'gamma': 0.99,  # 折扣率，控制未来奖励的重要性0.99
    'tau': 0.005,  # 软更新参数，控制目标网络的平滑更新
    'alpha': 0.2,  # 熵调节参数，控制策略的探索程度0.2
    'policy': 'Gaussian',  # 策略类型，可以是Gaussian或Deterministic策略
    'target_update_interval': 1,  # 目标网络更新的频率
    'automatic_entropy_tuning': True,  # 是否自动调节熵参数
    'cuda': True,
    'hidden_size': 512,  # Critic和Actor网络的隐藏层大小
    'lr': 0.0001,  # 学习率0.0001
    'buffer_capacity': 100000000,  # 添加buffer_capacity属性
    'seed': 145  # 添加seed属性
}

# 在这里初始化SAC代理
agent = SAC(state_dim, env.action_space, sac_args)

# 检查是否有预训练的模型参数
#pretrained_model_directory = 'sac_model'
#pretrained_model_episode = 1000   # 将这个 episode 修改为你想加载的模型参数的 episode

#if os.path.exists(pretrained_model_directory):
    #print(f"Loading pretrained model from episode {pretrained_model_episode}...")
    #agent.load_model(pretrained_model_directory, 'sac', pretrained_model_episode)
#else:
    #print("No pretrained model found. Starting from scratch.")

# 保存 Replay Buffer 的位置
buffer_save_path = 'checkpoints/sac_buffer_camera_control_.pkl'

# 尝试加载 Replay Buffer
if os.path.exists(buffer_save_path):
    print(f"Loading Replay Buffer from {buffer_save_path}...")
    agent.buffer.load_buffer(buffer_save_path)
else:
    print("No saved Replay Buffer found. Starting with an empty buffer.")

print(f"Buffer size: {len(agent.buffer)}")

# 存储每轮训练的奖励
episode_rewards = []

# 训练主循环
max_episodes = 1000 # 最大训练轮数

# 设置缓冲区初始样本数量
initial_buffer_size = 512
updates = 0

while len(agent.buffer) < initial_buffer_size:
    state = env.reset()
    done = False

    while not done:
        observation = state[:5]
        action = env.action_space.sample()  # 随机动作，用于填充缓冲区
        next_state, reward, done, _ = env.step(action)
        next_observation = next_state[:5]
        agent.buffer.push(observation, action, reward, next_observation, done)
        state = next_state

print(f"Initial buffer size: {len(agent.buffer)}")

for episode in range(max_episodes):
    observation = env.reset()
    # total_reward = 0
    # done = False
    total_reward = torch.zeros((batch_size,1),device="cuda")
    for _ in range(1000):
        #print(observation)

        action = agent.select_action(observation)
        #print(action)
        
        next_obs, reward, done, _ = env.step(action)
        # next_observation = next_state[:5]
        # agent.buffer.push(observation, action, reward, next_observation, done)
        total_reward += reward
        on_policy_memory = (observation, action, reward, next_obs, done)
        observation = next_obs

        # SAC算法中的训练步骤
        # qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha = agent.update_parameters(on_policy_memory, updates)
        agent.update_parameters(on_policy_memory, updates)
        updates += 1  # 每次更新后递增更新次数

    mean_reward = total_reward.mean()
    std_reward = total_reward.var()
    # episode_rewards.append(total_reward.mean())
    print(f"Episode: {episode + 1}, mean Reward: {mean_reward}, std reward{std_reward}")

# 关闭环境
env.close()

# 保存训练后的模型参数
save_dir = 'sac_model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

agent.save_model(save_dir, 'sac', max_episodes)

# 保存 Replay Buffer
agent.buffer.save_buffer('camera_control')