import numpy as np
import gym
from gym import spaces
import reco
import guidance
import json
from clamp_angle import clamp_angle
from PyQt5.QtCore import QObject, pyqtSignal
import torch
import warp as wp
vec5f = wp.types.vector(length=5, dtype=wp.float32)

FOCAL_MIN = 0.0043
FOCAL_MAX = 0.129
ELEVATION_MIN = 0.0 #np.deg2rad(0)
ELEVATION_MAX = 1.5707963267948966 #np.deg2rad(90)
D = 0.00818
ASPECT_RATIO = 16/9

class CameraControlEnv(gym.Env):
    def __init__(self, num_envs:int = 1024):
        super(CameraControlEnv, self).__init__()

        wp.init()
        # 定义连续动作空间，例如控制相机方位角、俯仰角和焦距的连续值
        self.state_dim = 9  # 新增：状态空间为11维
        self.observation_dim = 5
        self.action_dim = 3  # 动作维度为3维，包括方位角、俯仰角、焦距
        self.total_reward = 0


        # 状态空间的定义，包括方位角、俯仰角、焦距、gud_a、gud_e、速度和加速度
        self.observation_space = spaces.Box(
            low=torch.tensor([-np.pi, ELEVATION_MIN] + [FOCAL_MIN] + [-np.pi, 0]).numpy(),
            high=torch.tensor([np.pi, ELEVATION_MAX] + [FOCAL_MAX] + [np.pi, np.pi]).numpy(),
        ) 

        assert(self.observation_dim == self.observation_space.shape[0])
        # 新增：定义包含连续动作的连续动作空间，每个维度有相应的范围
        self.action_space = spaces.Box(
            low=np.array([-6, -6, -0.006], dtype=np.float32),
            high=np.array([6, 6, 0.006], dtype=np.float32),
            dtype=np.float32
        )

        self.camera_pos = np.array([0, 0, 0])

        self.blta = 0.000625
        self.start_point = np.array([900, 1200, 200])
        self.target_point = np.array([- 900, - 1200, 200])

        self.num_envs = num_envs
        self.device = "cuda"

    def reset(self, seed:int =227):
        
        # 构建文件的绝对路径
        file_path = 'tmpkw.json'

        # 读取 JSON 文件
        with open(file_path, "r") as file:
            data = json.load(file)
        
        # 打印输出 UAV 的数据
        uav_data = data.get("UAV", [])

        relative_trajectory = torch.tensor(uav_data, device="cuda") - torch.tensor(self.camera_pos, device="cuda")
        distance = torch.norm(relative_trajectory,dim=-1)
        
        elevation_rad = torch.arcsin(relative_trajectory[..., 2] / distance)
        azimuth_rad = torch.arctan2(relative_trajectory[..., 1], relative_trajectory[..., 0])

        self.uav_var_count = len(uav_data)

        self._target_info = wp.from_torch(torch.stack([distance, azimuth_rad, elevation_rad], dim=-1), dtype=wp.vec3f)
        self._agent_azimuth = torch.rand(size=(self.num_envs, ), device=self.device) * 2 * np.pi - np.pi
        self._agent_elevation = torch.rand(size=(self.num_envs, ), device=self.device) * np.pi

        self._agent_focal = torch.rand(size=(self.num_envs,), device=self.device) * (FOCAL_MAX - FOCAL_MIN) + FOCAL_MIN

        self._env_steps = torch.zeros_like(self._agent_focal, dtype=torch.int32)
        # self.agent_info = torch.concat([])

        # self._target_gud_a = torch.empty_like(self._agent_focal)
        # self._target_gud_e = torch.empty_like(self._agent_focal)

        action = self.action_space.sample()
        actions = torch.tensor(action, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        state,_,_,_ = self.step(actions)
        # self.x, self.y, self.z = self.trajectory[0] - self.camera_pos
        # self.done = False
        # self.current_t = 0
        # state = self.get_state()
        # self.total_reward = 0  # 重置累计奖励为0
        return state

    def step(self, action):
        delta_time = 1/60

        next_obs = torch.empty((self.num_envs, self.observation_dim), device=self.device)
        reward = torch.empty((self.num_envs, 1), device=self.device)
        dones = torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device)
        
        wp.launch(_step,
                  dim=self.num_envs,
                  inputs=[
                      wp.from_torch(self._agent_azimuth),
                      wp.from_torch(self._agent_elevation),
                      wp.from_torch(self._agent_focal),
                      
                      self._target_info,

                      wp.from_torch(action, dtype=wp.vec3),
                      delta_time,
                      wp.from_torch(next_obs, dtype=vec5f),
                      wp.from_torch(reward.squeeze(-1)),
                      wp.from_torch(dones.squeeze(-1)),
                      wp.from_torch(self._env_steps),
                      self.uav_var_count,
                      np.random.randint(200227),
                      self.blta])
        info = {}
        return next_obs, reward, dones, info

    # def get_state(self):
    #     # print(f"{self.x, self.y, self.z}")
    #     state = np.array([
    #         self.shared_data.azimuth,
    #         self.shared_data.elevation,
    #         self.shared_data.focal_length,
    #         self.shared_data.gud_a,  # 新增：gud_a
    #         self.shared_data.gud_e,  # 新增：gud_e
    #         self.x,
    #         self.y,
    #         self.z,
    #         self.current_t
    #     ])
    #     return state

@wp.func
def _clamp_azimuth(azimuth: float):
    if (azimuth > wp.pi):
        return azimuth - wp.tau
    elif (azimuth < -wp.pi):
        return  azimuth + wp.tau
    return azimuth

@wp.func
def _gen_ramdom_guide(azimuth: float,
                      elevation: float,
                      rnd_seed: wp.uint32):
    angle_offset1 = 2.0 * wp.randf(rnd_seed)
    angle_offset2 = 2.0 * wp.randf(rnd_seed)
    azimuth += angle_offset1 
    # azimuth = wp.round(azimuth, 2)
    elevation += angle_offset2
    # elevation = wp.round(elevation, 2)
    return wp.vec2f(azimuth, elevation)

@wp.func
def _normalize(value: float, 
               v_min: float,
               v_max: float)->float:
    return (value - v_min) / (v_max - v_min) * 2.0 - 1.0

@wp.func
def _get_observation(next_azimuth: float,
                     next_elevation: float,
                     next_focal: float,
                     azimuth: float,
                     elevation: float,
                     rnd_state: wp.uint32) -> vec5f:
    
    noise_guide_info = _gen_ramdom_guide(azimuth, elevation, rnd_state)

    NORMALIZE_OBS = False
    if NORMALIZE_OBS:
        normalized_azimuth = next_azimuth / wp.pi
        normalized_elevation = _normalize(next_elevation, ELEVATION_MIN, ELEVATION_MAX)
        normalized_focal = _normalize(next_focal, FOCAL_MIN, FOCAL_MAX)

        normalized_guide_azi = noise_guide_info[0] / wp.pi
        normalized_guide_ele = _normalize(noise_guide_info[1], ELEVATION_MIN, ELEVATION_MAX)

        return vec5f(normalized_azimuth,
                     normalized_elevation,
                     normalized_focal,
                     normalized_guide_azi,
                     normalized_guide_ele)
    else:        
        return vec5f(next_azimuth,
                     next_elevation,
                     next_focal,
                     noise_guide_info[0],
                     noise_guide_info[1])

RECOG_LEVEL3 = 3
RECOG_LEVEL2 = 2
RECOG_LEVEL1 = 1
RECOG_LEVEL0 = 0
RECOG_LEVEL_1 = -1

@wp.struct
class reward_state:
    reward: float
    state: int


@wp.func
def recog(distance: float, 
          focal_length: float):
    A = 0.6 # 无人机的尺寸为0.6*0.6
    U = 0.00000001668
    # 计算两点的间距
    d = (A * focal_length) / distance
    d = d*d
    n = d / U
    if n > 12:
        return RECOG_LEVEL3 # 辨认目标
    else:
        if n > 6:
            return RECOG_LEVEL2 # 识别目标
        else:
            if n > 1.5:
                return RECOG_LEVEL1 # 探测目标
            else:
                return RECOG_LEVEL0 # 未识别
            
@wp.func
def _get_reward_and_state(target_distance: float,
                          target_azimuth: float,
                          target_elevation: float,
                          next_azimuth: float,
                          next_elevation: float,
                          next_focal: float,
                          blta: float
                          ):
    fov_adjusted_horizontal = wp.atan(0.5 * D / next_focal)
    fov_adjusted_vertical = fov_adjusted_horizontal / ASPECT_RATIO

    gap_azi = wp.abs(target_azimuth - next_azimuth)
    gap_ele = wp.abs(target_elevation - next_elevation)

    reward_azi = (fov_adjusted_horizontal - gap_azi) / fov_adjusted_horizontal
    reward_ele = (fov_adjusted_vertical - gap_ele) / fov_adjusted_vertical

    reward_azi = wp.max(reward_azi, -1.0)
    reward_ele = wp.max(reward_azi, -1.0)

    reward0 = (reward_azi + reward_ele) / 2.0

    azi_low = next_azimuth - fov_adjusted_horizontal
    azi_high = next_azimuth + fov_adjusted_horizontal
    ele_low = next_elevation - fov_adjusted_vertical
    ele_high = next_elevation + fov_adjusted_vertical        

    azi_satisfied = azi_low <= target_azimuth <= azi_high 
    ele_satisfied = ele_low <= target_elevation <= ele_high

    g = RECOG_LEVEL_1
    if azi_satisfied and ele_satisfied:
        g = recog(target_distance, next_focal)
        if g == RECOG_LEVEL3:
            # 辨认目标
            # print('indentification')
            # print(f"{a1, a2, azimuth_deg}")
            reward1 = 1.0
            reward = blta * reward0 + reward1
        elif g == RECOG_LEVEL2:
            # 识别目标
            # print('recognition')
            # print(f"{a1, a2, azimuth_deg}")
            reward1 = 0.0
            reward = blta * reward0 + reward1
        elif g == RECOG_LEVEL1:
            # 探测目标
            # print('detection')
            # print(f"{a1, a2, azimuth_deg}")
            reward1 = 0.0
            reward = (blta * reward0 + reward1)
        elif g == RECOG_LEVEL1:
            # 未识别目标
            # print('undetection')
            reward1 = 0.0
            reward = blta * reward0 + reward1
    else:
        reward1 = 0.0
        reward = blta * reward0 + reward1
    
    
    return reward_state(reward, g)

@wp.kernel
def _step(azimuths : wp.array(dtype=float),
          elevations : wp.array(dtype=float),
          focals: wp.array(dtype=float),
          
          # TODO : multi-trajectories
          target_infos: wp.array(dtype=wp.vec3),
          
          actions: wp.array(dtype=wp.vec3),
          delta_time: float,
          next_observation: wp.array(dtype=vec5f),
          rewards: wp.array(dtype=float),
          dones: wp.array(dtype=int),
          steps: wp.array(dtype=int),
          time_limit: int,
          seed:int,
          
          lambda_reward_offset: float
          ):
    tid = wp.tid()
    
    current_azimuth = azimuths[tid]
    current_elevation = elevations[tid]
    next_focal = focals[tid]    
    target_info = target_infos[tid]

    # fetch action command
    current_action = actions[tid]
    current_action *= delta_time

    azimuth_inc, elevation_inc, focal_inc = current_action[0], current_action[1], current_action[2]

    # update state
    next_azimuth = _clamp_azimuth(current_azimuth + azimuth_inc)
    next_elevation = wp.clamp(current_elevation + elevation_inc, ELEVATION_MIN, ELEVATION_MAX)
    next_focal = wp.clamp(next_focal + focal_inc, FOCAL_MIN, FOCAL_MAX)

    # calculate reward and done
    target_distance, target_azimuth, target_elevation = target_info[0], target_info[1], target_info[2]

    reward_and_state = _get_reward_and_state(target_distance,
                                             target_azimuth,
                                             target_elevation,
                                             next_azimuth,
                                             next_elevation,
                                             next_focal,
                                             lambda_reward_offset)
    current_step = steps[tid] + 1

    done = current_step > time_limit or reward_and_state.state == RECOG_LEVEL3
    rewards[tid] = reward_and_state.reward

    rnd_state = wp.rand_init(seed, tid)

    if done:
        dones[tid] = 1

        current_step = 0
        # reset the environment
        next_azimuth = wp.randf(rnd_state) * wp.tau - wp.pi
        next_elevation = wp.randf(rnd_state) * wp.pi
        next_focal = wp.randf(rnd_state) * (FOCAL_MAX - FOCAL_MIN) + FOCAL_MIN
    else:
        dones[tid] = 0

    azimuths[tid] = next_azimuth
    elevations[tid] = next_elevation
    focals[tid] = next_focal
    steps[tid] = current_step

    # calculate obs
    next_observation[tid] = _get_observation(next_azimuth,
                                             next_elevation,
                                             next_focal,
                                             target_azimuth, 
                                             target_elevation,
                                             rnd_state)