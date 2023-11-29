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

class SharedData(QObject):
    azimuthChanged = pyqtSignal(int)
    elevationChanged = pyqtSignal(int)
    focalLengthChanged = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.azimuth = 0  # 方位角
        self.elevation = 45  # 仰角
        self.focal_length = 0.0043
        self.gud_a = 0  # 新增：gud_a，初始值为0
        self.gud_e = 0  # 新增：gud_e，初始值为0
        self.t = 0.5
        self.velocity = 25  # 新增：速度，初始值为0
        self.acceleration = 0  # 新增：加速度，初始值为0

    def setAzimuth(self, value):
        # 限制方位角在 -180 到 180 度之间
        self.azimuth = max(-180, min(value, 180))
        self.azimuthChanged.emit(self.azimuth)

    def setElevation(self, value):
        # 限制仰角在 -120 到 90 度之间
        self.elevation = max(-120, min(value, 90))
        self.elevationChanged.emit(self.elevation)

    def setFocalLength(self, value):
        # 限制焦距在 0.0043 到 0.129 之间
        self.focal_length = max(0.0043, min(value, 0.129))
        self.focalLengthChanged.emit(self.focal_length)

    # 新增：设置gud_a和gud_e的方法
    def setGudA(self, value):
        self.gud_a = value

    def setGudE(self, value):
        self.gud_e = value

    def setT(self, value):
        self.t = value

    # 新增：设置速度和加速度的方法
    def setVelocity(self, value):
        self.velocity = value

    def setAcceleration(self, value):
        self.acceleration = value

class BaseEnv(gym.Env):
    def __init__(self, path_to_trajectories:str):
        super().__init__()

        self.__trajectory_files = [os.path.join(path_to_trajectories, file) for file in os.listdir(path_to_trajectories)]
        assert(len(self.__trajectory_files) > 0, f"make sure there exists files in {path_to_trajectories}")

        self.observation_dim = 5
        self.action_dim = 3  # 动作维度为3维，包括方位角、俯仰角、焦距
        
        # 新增：定义包含连续动作的连续动作空间，每个维度有相应的范围
        self.action_space = spaces.Box(
            low=np.array([-6, -6, -0.006], dtype=np.float32),
            high=np.array([6, 6, 0.006], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
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
        with open(self.__trajectory_files[selected_index], "r") as file:
            data = json.load(file)
        self.__trajectory = data["UAV"]
        self.__uav_var_count = len(self.__trajectory)
        
        self.__current_t = 0
        self.shared_data.azimuth = 0
        self.shared_data.elevation = 45
        self.shared_data.focal_length = 0.0043
        self.shared_data.gud_a = 0  # 新增：重置gud_a为0
        self.shared_data.gud_e = 0  # 新增：重置gud_e为0

        self.shared_data.velocity = 25  # 新增：重置速度为max_speed
        self.shared_data.acceleration = 0  # 新增：重置加速度为0
    
    def update_env(self, action):
        self.__current_t += 1
        
        self.__update_defender_info(action)
        self.__update_uav_info()
         
    def is_done(self):
        "Return `true` if we reach to the end of this episode."
        return self.__current_t >= self.__uav_var_count
    
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
        
    def __update_uav_info(self):
        current_frame = self.get_current_frame()
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
            self.shared_data.azimuth,
            self.shared_data.elevation,
            self.shared_data.focal_length,
            self.shared_data.gud_a,  # 新增：gud_a
            self.shared_data.gud_e,  # 新增：gud_e
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
        
        reward, reward1 = self.calculate_reward()
        obs = self.get_observation()
        done = self.is_done() or reward1 == 100
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

        self.shared_data.gud_e = self.shared_data.gud_e
        self.shared_data.gud_a = self.shared_data.gud_a
        # print(f"{self.gud_a, self.gud_e, azimuth_deg, elevation_deg}")

        # 逐步逼近奖励
        azimuth_diff1 = np.abs(azimuth - self.shared_data.gud_a)
        elevation_diff1 = np.abs(elevation - self.shared_data.gud_e)
        if azimuth_diff1 <= 5 and elevation_diff1 <= 5:
            # print('correct', end=' ')
            reward0 = 0
        else:
            # print('false', end=' ')
            reward0 = 0
            
        capture_state = self.get_capture_state()

        if focal_length == 0.0043:
            reward2 = 1
        else:
            reward2 = 0

        # 辨认奖励的计算
        if (capture_state == CaptureState.identifying_target):
            # 辨认目标
            # print('indentification')
            # print(f"{a1, a2, azimuth_deg}")
            reward1 = 100
            self.trigger_condition = True
            reward = reward0 + reward1 + reward2
            return reward, reward1
        elif (capture_state == CaptureState.recognizing_target):
            # 识别目标
            # print('recognition')
            # print(f"{a1, a2, azimuth_deg}")
            reward1 = 50
            reward = reward0 + reward1 + reward2
            return reward, reward1
        elif (capture_state == CaptureState.detecting_target):
            # 探测目标
            # print('detection')
            # print(f"{a1, a2, azimuth_deg}")
            reward1 = 20
            reward = reward0 + reward1 + reward2
            return reward, reward1
        elif(capture_state == CaptureState.out_of_focal):
            reward1 = 0
            reward = reward0 + reward1 + reward2
            return reward, reward1
        elif(capture_state == CaptureState.out_of_horizontal_or_vertial):
            reward1 = 0
            reward = reward0 + reward1 + reward2
            return reward, reward1
        else:
            raise NotImplementedError