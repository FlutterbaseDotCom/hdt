import wandb
import random

import torch
import torch.nn as nn
from torchviz import make_dot
from stable_baselines3 import DQN
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


env =  gym.make('CarRacing-v2', continuous=False) #, render_mode='human'


model = DQN
tmp_model_path ='./models/dql_pretrained/dql_rl_11.zip'
loaded_model = model.load(tmp_model_path)



wrapped_env = VecTransposeImage(DummyVecEnv([lambda: env]))
fx = NatureCNN(wrapped_env.observation_space, features_dim=512)
fx.load_state_dict(torch.load('./models/cnn_feature_extractor/nature_cnn_checkpoint.pth'))

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



# play model
obs = env.reset()
if len(obs) == 2:
    obs = obs[0]
with torch.no_grad():
    for _ in range(5):
        action, _states = loaded_model.predict(obs,deterministic=True)
        res = fx(prepare_observation_tensor(obs))
        print('#################')
        print('hidden state:')
        print(res.shape)
        print(res)
        obs, reward, done, t, i = env.step(action)





print(loaded_model.policy)


env.close()