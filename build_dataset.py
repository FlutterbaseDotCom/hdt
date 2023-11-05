import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from dt.configuration_decision_transformer import DecisionTransformerConfig
from dt.modeling_decision_transformer import DecisionTransformerModel


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

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from dt.configuration_decision_transformer import DecisionTransformerConfig
from dt.modeling_decision_transformer import DecisionTransformerModel
from extract_cnn import prepare_observation_array
from dt.trainable_dt import DecisionTransformerGymDataCollator, TrainableDT

os.environ["WANDB_DISABLED"] = "true" # we diable weights and biases logging for this tutorial



env =  gym.make('CarRacing-v2', continuous=False) #, render_mode='human'


model = DQN
tmp_model_path ='./models/dql_pretrained/dql_rl_11.zip'
loaded_model = model.load(tmp_model_path)


NUM_EPISODES = 100
features = {
    "observations": [],
    "actions": [],
    "rewards": [],
    "dones": [],
}
for episode in range(NUM_EPISODES):
    print(f"Episode: {episode} of {NUM_EPISODES}:" )
    [obs, _] = env.reset()
    done = False

    o, a, r, d = [], [], [], []
    total_reward = 0
    sti = 0
    tmp_max_sti = 1000
    while not done:
        sti = sti + 1
        if sti > tmp_max_sti:
            break

        # if random.random() < epsilon:
        #     action = 3# env.action_space.sample()
        # else:
        action, _states = loaded_model.predict(obs,deterministic=True)
        new_obs, reward, done, t, i = env.step(action)
        total_reward = total_reward + reward
        oarr = prepare_observation_array(obs)
        o.append(oarr.flatten())
        a.append(action.item())
        r.append(reward)
        d.append(done)
        obs = new_obs
        print(".", end="")

        # check if last 50 steps does not contain a single positive reward
        if len(r) > 100 and max(r[-50:]) <= 0:
            # cut last 50 and set done to True
            r = r[:-50]
            d = d[:-50]
            a = a[:-50]
            d[-1] = True
            print('stopping die to the last 50 steps not negative rewards')
            break
    print(f"Total reward: {total_reward}")
    features["observations"].append(o)
    features["actions"].append(a)
    features["rewards"].append(r)
    features["dones"].append(d)

env.close()
print(len(features["actions"]))

dataset = Dataset.from_dict(features)
dataset.save_to_disk('datasets/car_racing_sm/')

#dataset = dataset.train_test_split(test_size=0.1)
# dataset = {
#     "train": dataset
# }


# collator = DecisionTransformerGymDataCollator(dataset["train"])

# config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
# model = TrainableDT(config)


# training_args = TrainingArguments(
#     output_dir="output/",
#     remove_unused_columns=False,
#     num_train_epochs=120,
#     per_device_train_batch_size=64,
#     learning_rate=1e-4,
#     weight_decay=1e-4,
#     warmup_ratio=0.1,
#     optim="adamw_torch",
#     max_grad_norm=0.25,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     data_collator=collator,
# )

# trainer.train()
