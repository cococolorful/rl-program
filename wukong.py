from typing import Any, List, Union, Sequence, Optional
import copy
import numpy as np
import gym

from ding.envs import BaseEnv, BaseEnvTimestep, update_shape
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray, to_list
from env import CameraControlEnv
from ding.envs import ObsPlusPrevActRewWrapper


class AtariEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        
        self.reset()
        
    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._make_env()
            if self._replay_path is not None:
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )
            if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
                self._env = ObsPlusPrevActRewWrapper(self._env)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._eval_episode_return = 0.
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)

        obs, rew, done, info = self._env.step(action)
        
        return BaseEnvTimestep(obs, rew, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def _make_env(self):
        return CameraControlEnv("trajectories")

    def __repr__(self) -> str:
        return "DI-engine Atari Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]


from ditk import logging
from ding.model import QAC
from ding.policy import SACPolicy
from ding.envs import BaseEnvManagerV2, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext,OfflineRLContext
from ding.framework.middleware import data_pusher, StepCollector, interaction_evaluator, \
    CkptSaver, OffPolicyLearner, termination_checker, online_logger
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from dizoo.classic_control.pendulum.envs.pendulum_env import PendulumEnv
from wukong_config import main_config, create_config
import torch
from ding.envs import BaseEnvManager, DingEnvWrapper,SyncSubprocessEnvManager,AsyncSubprocessEnvManager
from easydict import EasyDict

    
def main(eval_only:bool = False):
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    

    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    model = QAC(**cfg.policy.model)
    buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
    policy = SACPolicy(cfg.policy, model=model)
    
    buffer_.push()
    
    if eval_only:
        evaluator_env = BaseEnvManager(env_fn=[lambda : DingEnvWrapper(
            AtariEnv(cfg.env),
            EasyDict(env_wrapper='default'),
        ) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager)
            
        policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))
        evaluator = InteractionSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, exp_name=cfg.exp_name
        )
        evaluator.eval()
    else:
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda : DingEnvWrapper(
            AtariEnv(cfg.env),
            EasyDict(env_wrapper='default'),) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda : DingEnvWrapper(
            AtariEnv(cfg.env),
            EasyDict(env_wrapper='default'),) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )
    
        with task.start(async_mode=False, ctx=OnlineRLContext()):    
            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(
                StepCollector(cfg, policy.collect_mode, collector_env, random_collect_size=cfg.policy.random_collect_size)
            )
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(CkptSaver(policy, cfg.exp_name, train_freq=100_000))
            task.use(termination_checker(max_train_iter=cfg.env.stop_value))
            task.use(online_logger())
            task.run()


if __name__ == "__main__":
    main(eval_only=False)
    main(eval_only=True)
