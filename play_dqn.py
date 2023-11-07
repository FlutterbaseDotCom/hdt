
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


model = DQN
tmp_model_path ='./models/dql_pretrained/dql_rl_11.zip'
loaded_model = model.load(tmp_model_path)

def prepare_observation_tensor( observation):
    observation_space_shape = (3,96,96)
    if not (observation.shape == observation_space_shape or observation.shape[1:] == observation_space_shape):
        # Try to re-order the channels
        transpose_obs = VecTransposeImage.transpose_image(observation)
        if transpose_obs.shape == observation_space_shape or transpose_obs.shape[1:] == observation_space_shape:
            observation = transpose_obs
    # Add batch dimension if needed
    observation = observation.reshape((-1, *observation_space_shape))

    #convert to tensor
    observation = torch.as_tensor(observation, device='cpu') #cpu!!!
    observation = observation.float() / 255.0
    return observation

env = gym.make('CarRacing-v2', continuous=False, render_mode='human')
print("Observation space: ", env.observation_space)
print("Action space: ", env.action_space)
# play model
obs = env.reset()
if len(obs) == 2:
    obs = obs[0]
done = False
r = []
with torch.no_grad():
    while not done:
        action, _states = loaded_model.predict(obs,deterministic=True)
        obs, reward, done, t, i = env.step(action)
        r.append(reward)
        if len(r) > 100 and max(r[-100:]) <= 0:
            print('Terminated because of consecutive 100 steps without reward')
            break


env.close()
