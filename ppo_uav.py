import gymnasium as gym
import os
from stable_baselines3  import   PPO,SAC
from env import CameraControlEnv  # CameraControlEnv是你的自定义环境
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# env = CameraControlEnv(path_to_trajectories="trajectories")
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
import random
import math
import pickle
# Replay Memory
class ReplayBuffer:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, next_state, action, reward,  done

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
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.k = 0
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if self.num_timesteps % 3000 == 0:
            model = self.model
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=3)
            self.logger.record("mean_reward", mean_reward)
            self.logger.record("std_reward", std_reward)
        return True
    def _on_training_end(self) -> None:
        super()._on_training_end()
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        self.logger.record("mean_reward", mean_reward)
        self.logger.record("std_reward", std_reward)

class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str = None,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        self.buffer = ReplayBuffer(512, 512)
        self.buffer.load_buffer("checkpoints/sac_buffer_camera_control_.pkl")
        self.first = True
    def _on_training_start(self) -> None:
        super()._on_training_start()
        self.save_path = self.logger.get_dir()
        
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")
    def on_rollout_end(self) -> None:
        if self.first:
            self.model.replay_buffer.reset()
            
            self.model.replay_buffer.handle_timeout_termination = False
            [self.model.replay_buffer.add(*self.buffer.sample(1),infos=None) for _ in range(512)]
            self.first = False
            a = 1
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True

model_type = SAC
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=3000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)
vec_env = DummyVecEnv([lambda: CameraControlEnv(path_to_trajectories="trajectories")])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
model = model_type("MlpPolicy", vec_env, verbose=1,tensorboard_log=f"./garbage/{str(model_type).split('.')[-2]}/")
model.learn(total_timesteps=6_000_000, progress_bar=True, callback=[TensorboardCallback(),checkpoint_callback])

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = model.logger.get_dir()
model_path = os.path.join(log_dir, "model.pkl")
env_path = os.path.join(log_dir, "vec_normalize.pkl")
model.save(model_path)
vec_env.save(env_path)

del model
del vec_env
vec_env = DummyVecEnv([lambda: CameraControlEnv(path_to_trajectories="trajectories")])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
vec_env = VecNormalize.load(env_path,vec_env)
model = model_type.load(model_path, env=vec_env)
vec_env = model.get_env()

obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

vec_env.close()