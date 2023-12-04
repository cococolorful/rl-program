import numpy as np
import gym
from gym import spaces
import reco
import guidance
import json
import os
from reco import CaptureState, assit
from clamp_angle import clamp_angle
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], 
                "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
class BaseEnv(gym.Env):
    def __init__(self, path_to_trajectories:str):
        super().__init__()

        self.__trajectory_files = [os.path.join(path_to_trajectories, file) for file in os.listdir(path_to_trajectories)]
        # assert(len(self.__trajectory_files) > 0, f"make sure there exists files in {path_to_trajectories}")

        self.observation_dim = 5
        self.action_dim = 3  # 动作维度为3维，包括方位角、俯仰角、焦距
        
        # 新增：定义包含连续动作的连续动作空间，每个维度有相应的范围
        self.action_space = spaces.Box(
            low=np.array([-6, -6, -0.006], dtype=np.float32),
            high=np.array([6, 6, 0.006], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.observation_dim,), dtype=np.float32
        )
        
        self.shared_data = SharedData()
        self.CAMERA_POS = np.array([0, 0, 0])
        self.D = 0.00818
        self.ASPECT_RATIO = 16/9
        
    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        # random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    def reset(self):
        # Generate the trajectory
        selected_index = np.random.randint(len(self.__trajectory_files))
        self.load_trajectory(self.__trajectory_files[selected_index])
        self.__uav_var_count = len(self.__trajectory)
        
        self.__current_t = 0
        self.shared_data.azimuth = 0
        self.shared_data.elevation = 45
        self.shared_data.focal_length = 0.0043
        self.shared_data.gud_a = 0  # 新增：重置gud_a为0
        self.shared_data.gud_e = 0  # 新增：重置gud_e为0

        self.shared_data.velocity = 25  # 新增：重置速度为max_speed
        self.shared_data.acceleration = 0  # 新增：重置加速度为0
        
        self.__update_uav_info(0)
    
    def load_trajectory(self, path_to_trajectory):
        with open(path_to_trajectory, "r") as file:
            data = json.load(file)
        self.__trajectory = data["UAV"]
    
    def update_env(self, action):
        self.__current_t += 1           # TODO:
        
        self.__update_defender_info(action)
        self.__update_uav_info(self.__current_t)
         
    def is_done(self):
        "Return `true` if we reach to the end of this episode."
        return self.__current_t >= self.__uav_var_count - 1
    
    def get_capture_state(self):
        
        return CaptureState.get_capture_state(self.defender_azimuth_low, self.defender_azimuth_high, self.__azimuth_deg,
                                              self.defender_elevation_low, self.defender_elevation_high, self.__elevation_deg,
                                              self.__distance, self.shared_data.focal_length)
    
    def get_current_frame(self):
        return self.__current_t - 1
        
    def __update_defender_info(self, action):
        azimuth_1, elevation_1, focal_length1 = action

        # 获取上一个状态的azimuth和elevation
        prev_azimuth = self.shared_data.azimuth
        prev_elevation = self.shared_data.elevation
        prev_focal_length = self.shared_data.focal_length
        
        # 计算azimuth和elevation相对于上一个状态的变化
        azimuth = clamp_angle(prev_azimuth + azimuth_1)
        elevation = np.clip(prev_elevation + elevation_1, -120, 90)
        focal_length = np.clip(prev_focal_length + focal_length1, 0.0043, 0.129)
        
        # 相机的水平视场角度
        half_fov_adjusted_horizontal = np.rad2deg(np.arctan(0.5 * self.D / focal_length))
        half_fov_adjusted_horizontal = round(half_fov_adjusted_horizontal, 2)
        # 相机的上下俯仰视场角度
        half_fov_adjusted_vertical = half_fov_adjusted_horizontal / self.ASPECT_RATIO
        half_fov_adjusted_vertical = round(half_fov_adjusted_vertical, 2)
        
        # 将动作应用于相机的方位角、俯仰角和焦距
        self.shared_data.azimuth = azimuth
        self.shared_data.elevation = elevation
        self.shared_data.focal_length = focal_length
        
        self.defender_azimuth_low = azimuth - half_fov_adjusted_horizontal
        self.defender_azimuth_high = azimuth + half_fov_adjusted_horizontal
        self.defender_elevation_low = elevation - half_fov_adjusted_vertical
        self.defender_elevation_high = elevation + half_fov_adjusted_vertical
        
    def __update_uav_info(self, current_frame):
        x, y, z = self.__trajectory[current_frame]
        
        # 计算目标相对于相机的位置矢量
        target_vector = np.array([x, y, z]) - self.CAMERA_POS

        # 计算俯角和仰角
        distance = np.linalg.norm(target_vector)
        elevation_rad = np.arcsin(target_vector[2] / distance)
        azimuth_rad = np.arctan2(target_vector[1], target_vector[0])

        elevation_deg = np.rad2deg(elevation_rad)
        azimuth_deg = np.rad2deg(azimuth_rad)
        
        # 对真实方位信息模糊后获得指引信息
        self.__x, self.__y, self.__z = x,y,z
        self.__distance = distance
        self.__elevation_deg = round(elevation_deg, 2)
        self.__azimuth_deg = round(azimuth_deg, 2)

        self.shared_data.gud_e, self.shared_data.gud_a = guidance.gd(self.__elevation_deg, self.__azimuth_deg)
        
    def __get_state(self):
        # print(f"{self.x, self.y, self.z}")
        state = np.array([
            self.shared_data.azimuth / 180,
            self.shared_data.elevation / 90,
            ((self.shared_data.focal_length - 0.0043) / (0.129 - 0.0043) -0.5) * 2,
            self.shared_data.gud_a / 180,  # 新增：gud_a
            self.shared_data.gud_e / 90,  # 新增：gud_e
            self.shared_data.velocity,  # 新增：速度
            self.shared_data.acceleration,  # 新增：加速度
            self.__x,           
            self.__y,
            self.__z,
            self.__current_t
        ], dtype=np.float32)
        return state
class CameraControlEnv(BaseEnv):
    def __init__(self, path_to_trajectories:str):
        if path_to_trajectories is None:
            raise RuntimeError
        super().__init__(path_to_trajectories)

        # 定义连续动作空间，例如控制相机方位角、俯仰角和焦距的连续值
        self.state_dim = 11  # 新增：状态空间为11维

        self.total_reward = 0

        self.trigger_condition = False
        self.reward_compensation = 0  # 初始奖励补偿
        self.compensation_decay = 0.99  # 每次递减的因子

        # 状态空间的定义，包括方位角、俯仰角、焦距、gud_a、gud_e、速度和加速度
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )        

    def reset(self):
        super().reset()
        
        self.trigger_condition = False
        self.done = False
        self.total_reward = 0  # 重置累计奖励为0
        
        return self.get_observation()

    def step(self, action):
        self.update_env(action)
        
        reward = self.calculate_reward()
        obs = self.get_observation()
        done = self.is_done() or reward == 100
        # print(f"{self.current_t, done}")
        if self.trigger_condition:
            reward += self.reward_compensation
            # 逐渐减小奖励补偿
            self.reward_compensation *= self.compensation_decay
        self.total_reward += reward
        info = {}
        return obs, reward, done, info
    
    def get_observation(self):
        return self._BaseEnv__get_state()[:5]
        
    def calculate_reward(self):
        azimuth = self.shared_data.azimuth
        elevation = self.shared_data.elevation
        focal_length = self.shared_data.focal_length

        # print(f"{self.gud_a, self.gud_e, azimuth_deg, elevation_deg}")
        gud_a,gud_e = self.shared_data.gud_a,self.shared_data.gud_e
        # # 逐步逼近奖励
        # azimuth_diff1 = np.abs(azimuth - self.shared_data.gud_a)
        # elevation_diff1 = np.abs(elevation - self.shared_data.gud_e)
        # if azimuth_diff1 <= 5 and elevation_diff1 <= 5:
        #     # print('correct', end=' ')
        #     reward0 = 1
        # else:
        #     # print('false', end=' ')
        #     reward0 = -100# -((azimuth_diff1 - 5) + (elevation_diff1 - 5))
        uav_dir = np.array([np.sin(gud_e) * np.cos(gud_a),
                            np.sin(gud_e) * np.sin(gud_a),
                            np.cos(gud_e)])
        defender_dir = np.array([np.sin(elevation) * np.cos(azimuth),
                            np.sin(elevation) * np.sin(azimuth),
                            np.cos(elevation)])
        
        capture_state = self.get_capture_state()

        # 辨认奖励的计算
        if (capture_state == CaptureState.identifying_target):
            # 辨认目标
            return 100000
        elif (capture_state == CaptureState.recognizing_target):
            # 识别目标
            return 50
        elif (capture_state == CaptureState.detecting_target):
            # 探测目标
            return 20
        elif(capture_state == CaptureState.out_of_focal):
            return 3
        elif(capture_state == CaptureState.out_of_horizontal_or_vertial):
            return -(uav_dir*defender_dir).sum()
        else:
            raise NotImplementedError