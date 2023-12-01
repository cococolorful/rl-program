import torch
import traceback
import socket
import json

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(3)
    except Exception as inst:
        pass
            
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(str):
    global conn
    if str != None:
        message_bytes = bytes(json.dumps(send_data),'ascii')
        conn.sendall(len(message_bytes).to_bytes(4, 'little'))
        conn.sendall(message_bytes)
    # conn.sendall(bytes(verify, 'ascii'))

def receive():
    message = read()
    input = torch.tensor(message["input"])
    return input




import gymnasium as gym
import os
from stable_baselines3  import   PPO,SAC
from sac.SAC_1.env import CameraControlEnv  # CameraControlEnv是你的自定义环境
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# env = CameraControlEnv(path_to_trajectories="trajectories")
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback

model_type = PPO


# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "sac\SAC_3"
log_dir = "ppo\PPO_1"
index = 4650000
model_path = os.path.join(log_dir, f"rl_model_{index}_steps.zip")
env_path = os.path.join(log_dir, f"rl_model_replay_buffer_{index}_steps.pkl")



vec_env = DummyVecEnv([lambda: CameraControlEnv(path_to_trajectories="trajectories")])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
# vec_env = VecNormalize.load(env_path,vec_env)
model = model_type.load(model_path, env=vec_env)
vec_env = model.get_env()



import random

def gd(ele, azi):
    angle_offset1 = 0#5 * random.uniform(0, 1)
    angle_offset2 = 0#3 * random.uniform(0, 1)
    azimuth = angle_offset1 + azi
    azimuth = round(azimuth, 2)
    elevation = angle_offset2 + ele
    elevation = round(elevation, 2)
    return(elevation, azimuth)

obs = vec_env.reset()

init("127.0.0.1",6009)
while True:
    if conn == None:
        try_connect()
    while conn != None:
        try:
            input = receive()

            gd_elevation, gd_azimuth = gd(input[4].item(), input[3].item())
            observation = input
            observation[4],observation[3] = gd_elevation, gd_azimuth
            observation[2] /= 1000

            # print(f"environ is:{observation}")
            action, _states = model.predict(obs, deterministic=True)
            action = action.flatten()
            action[2] *= 1000
            # action = action
            # action[0], action[1] = gd_elevation, gd_azimuth
            # action[:2] = np.clip(action[:2],-1,1)
            # print(f"action is:{action} \n")
            # action = torch.tensor([action[2],action[0],action[2]])
            send_data = {}
            send_data['action'] = action.tolist()
            send(json.dumps(send_data))
            break
        except Exception as e:
            conn = None