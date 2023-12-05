import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random
import math
import pickle
import gymnasium as gym
from gym import spaces as spaces
from torch.distributions import Normal
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
# 辅助函数
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# import torch
# class GPURolloutBuffer:
#     """
#     Rollout buffer used in on-policy algorithms like A2C/PPO.
#     It corresponds to ``buffer_size`` transitions collected
#     using the current policy.
#     This experience will be discarded after the policy update.
#     In order to use PPO objective, we also store the current value of each state
#     and the log probability of each taken action.

#     The term rollout here refers to the model-free notion and should not
#     be used with the concept of rollout used in model-based RL or planning.
#     Hence, it is only involved in policy and value function training but not action selection.

#     :param buffer_size: Max number of element in the buffer
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param device: PyTorch device
#     :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
#         Equivalent to classic advantage when set to 1.
#     :param gamma: Discount factor
#     :param n_envs: Number of parallel environments
#     """

#     observations: torch.Tensor
#     actions: torch.Tensor
#     rewards: torch.Tensor
#     advantages: torch.Tensor
#     returns: torch.Tensor
#     episode_starts: torch.Tensor
#     log_probs: torch.Tensor
#     values: torch.Tensor

#     def __init__(
#         self,
#         buffer_size: int,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         device: Union[torch.device, str] = "auto",
#         gae_lambda: float = 1,
#         gamma: float = 0.99,
#         n_envs: int = 1,
#     ):
#         # super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
#         self.gae_lambda = gae_lambda
#         self.gamma = gamma
#         self.generator_ready = False
#         self.reset()

#     def reset(self) -> None:
#         self.observations = torch.empty((self.buffer_size, self.n_envs, *self.obs_shape), dtype=torch.float32, device=self.device)
#         self.actions = torch.empty((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.float32, device=self.device)
#         self.rewards = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
#         self.returns = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
#         self.episode_starts = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
#         self.values = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
#         self.log_probs = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
#         self.advantages = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
#         self.generator_ready = False
#         super().reset()

#     def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
#         """
#         Post-processing step: compute the lambda-return (TD(lambda) estimate)
#         and GAE(lambda) advantage.

#         Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
#         to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
#         where R is the sum of discounted reward with value bootstrap
#         (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

#         The TD(lambda) estimator has also two special cases:
#         - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
#         - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

#         For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

#         :param last_values: state value estimation for the last step (one for each env)
#         :param dones: if the last step was a terminal step (one bool for each env).
#         """
#         # Convert to numpy
#         # last_values = last_values.clone().cpu().numpy().flatten()
#         last_values = last_values.flatten()

#         last_gae_lam = 0
#         for step in reversed(range(self.buffer_size)):
#             if step == self.buffer_size - 1:
#                 next_non_terminal = 1.0 - dones
#                 next_values = last_values
#             else:
#                 next_non_terminal = 1.0 - self.episode_starts[step + 1]
#                 next_values = self.values[step + 1]
#             delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
#             last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
#             self.advantages[step] = last_gae_lam
#         # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
#         # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
#         self.returns = self.advantages + self.values

#     def add(
#         self,
#         obs: torch.Tensor,
#         action: torch.Tensor,
#         reward: np.ndarray,
#         episode_start: np.ndarray,
#         value: th.Tensor,
#         log_prob: th.Tensor,
#     ) -> None:
#         """
#         :param obs: Observation
#         :param action: Action
#         :param reward:
#         :param episode_start: Start of episode signal.
#         :param value: estimated value of the current state
#             following the current policy.
#         :param log_prob: log probability of the action
#             following the current policy.
#         """
#         if len(log_prob.shape) == 0:
#             # Reshape 0-d tensor to avoid error
#             log_prob = log_prob.reshape(-1, 1)

#         # Reshape needed when using multiple envs with discrete observations
#         # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
#         if isinstance(self.observation_space, spaces.Discrete):
#             raise NotImplementedError
#             obs = obs.reshape((self.n_envs, *self.obs_shape))

#         # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
#         action = action.reshape((self.n_envs, self.action_dim))
        
#         self.observations[self.pos] = obs
#         self.actions[self.pos] = action
#         self.rewards[self.pos] = reward
#         self.episode_starts[self.pos] = episode_start if isinstance(episode_start, torch.Tensor) else torch.from_numpy(episode_start).to(self.device)
#         self.values[self.pos] = value.flatten()
#         self.log_probs[self.pos] = log_prob
#         self.pos += 1
#         if self.pos == self.buffer_size:
#             self.full = True

#     def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
#         assert self.full, ""
#         indices = torch.randperm(self.buffer_size * self.n_envs)
#         # Prepare the data
#         if not self.generator_ready:
#             _tensor_names = [
#                 "observations",
#                 "actions",
#                 "values",
#                 "log_probs",
#                 "advantages",
#                 "returns",
#             ]

#             for tensor in _tensor_names:
#                 self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
#             self.generator_ready = True

#         # Return everything, don't create minibatches
#         if batch_size is None:
#             batch_size = self.buffer_size * self.n_envs

#         start_idx = 0
#         while start_idx < self.buffer_size * self.n_envs:
#             yield self._get_samples(indices[start_idx : start_idx + batch_size])
#             start_idx += batch_size

#     def _get_samples(
#         self,
#         batch_inds: np.ndarray,
#         env,
#     ) :
#         data = (
#             self.observations[batch_inds],
#             self.actions[batch_inds],
#             self.values[batch_inds].flatten(),
#             self.log_probs[batch_inds].flatten(),
#             self.advantages[batch_inds].flatten(),
#             self.returns[batch_inds].flatten(),
#         )
#         return RolloutBufferSamples(*tuple(map(self.to_torch, data)))



# Replay Memory
class ReplayBuffer:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        # self.observations = torch.empty((self.buffer_size, self.n_envs, *self.obs_shape), dtype=torch.float32, device=self.device)
        # self.actions = torch.empty((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.float32, device=self.device)
        # self.rewards = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        # self.returns = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        # self.episode_starts = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        # self.values = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        # self.log_probs = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        # self.advantages = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)


    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}.pkl".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

# SAC算法
class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.updates = 0  # 初始化更新次数为0
        self.gamma = args['gamma'] # 折扣率，控制未来奖励的重要性
        self.tau = args['tau'] # 软更新参数，控制目标网络的平滑更新
        self.alpha = args['alpha'] # 熵调节参数，控制策略的探索程度
        self.policy_type = args['policy'] # 策略类型，可以是Gaussian或Deterministic策略
        self.target_update_interval = args['target_update_interval'] # 目标网络更新的频率
        self.automatic_entropy_tuning = args['automatic_entropy_tuning'] # 是否自动调节熵参数
        self.device = torch.device("cuda" if args['cuda'] else "cpu")
        self.buffer = ReplayBuffer(args['buffer_capacity'], args['seed'])
        hidden_size = args['hidden_size'] # Critic和Actor网络的隐藏层大小
        lr = args['lr'] # 学习率

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr) # 价值网络的优化器定义

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr) # 温度参数以及优化器定义

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device) # 创建Actor网络
            self.policy_optim = Adam(self.policy.parameters(), lr=lr) # 策略优化器定义

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, evaluate=False): # 输出随机策略动作
        # state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
            return action
        # return action.detach().cpu().numpy()[0]

    def update_parameters(self, on_policy_memory, updates):
        # Sample a batch from memory
        # 对价值网络进行训练
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = on_policy_memory

        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # 对策略网络进行训练
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # 温度参数自适应调节
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        # 目标网络参数软更新
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        return 
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    # 保存模型参数
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # 加载模型参数
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

    # 保存模型参数
    def save_model(self, save_dir, env_name, episode):
        suffix = f"episode_{episode}"
        ckpt_path = os.path.join(save_dir, f"sac_checkpoint_{env_name}_{suffix}")
        self.save_checkpoint(env_name, suffix, ckpt_path)

    # 加载模型参数
    def load_model(self, save_dir, env_name, episode, evaluate=False):
        suffix = f"episode_{episode}"
        ckpt_path = os.path.join(save_dir, f"sac_checkpoint_{env_name}_{suffix}")
        self.load_checkpoint(ckpt_path, evaluate)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# 定义Soft Q-network，价值网络定义
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):# 定义前向传播函数
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class GaussianPolicy(nn.Module):# Actor网络定义
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions) 

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):# 前向传播函数定义
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):# 动作采样以及概率计算函数
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)