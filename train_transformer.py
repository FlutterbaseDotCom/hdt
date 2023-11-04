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
from datasets import load_from_disk
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from dt.configuration_decision_transformer import DecisionTransformerConfig
from dt.modeling_decision_transformer import DecisionTransformerModel
from extract_cnn import prepare_observation_array
from trainable_dt import DecisionTransformerGymDataCollator, TrainableDT
import toml
from colabgymrender.recorder import Recorder

# TOML-formatted string
config_toml = """
PREFIX              = 'DT'
LOG_INTERVAL        = 1
save_steps          = 5
num_train_epochs    = 5
per_device_train_batch_size=64
learning_rate       = 0.0001
weight_decay        = 0.0001
warmup_ratio        = 0.1
max_grad_norm       = 0.25
"""

config_toml = toml.loads(config_toml)

LOAD_SAVED_MODEL    = False

RUN_NUM = 13
WANDB_ID            = "dt_rl_"+str(RUN_NUM)
WNDB_NAME           = "DT_RL_"+str(RUN_NUM)
MODEL_SAVE_NAME     = WNDB_NAME
SAVED_MODEL_VERSION = "latest"



import os

os.environ['WANDB_NOTEBOOK_NAME'] = 'DT.ipynb'
os.environ['WANDB_MODE']='online'
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_DISABLED"] = "false" # we diable weights and biases logging for this tutorial


wandb.init(resume=WANDB_ID,
           mode="online",
           entity="yakiv",
            project="CarRacingDT",
            #resume= "allow"
            config=config_toml
           )
wandb.run.name = WNDB_NAME

env =  gym.make('CarRacing-v2', continuous=False) #, render_mode='human'

dataset = load_from_disk('datasets/car_racing_sm/')
#dataset = dataset.map(lambda x: x)

#dataset = dataset.train_test_split(test_size=0.1)
dataset = {
    "train": dataset
}


collator = DecisionTransformerGymDataCollator(dataset["train"])

config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
print(config.to_dict())

model = TrainableDT(config)


training_args = TrainingArguments(
    output_dir="output/",
    report_to="wandb",  
    save_steps=config_toml["save_steps"],
    remove_unused_columns=False,
    optim="adamw_torch",
    num_train_epochs=config_toml["num_train_epochs"],
    per_device_train_batch_size=config_toml["per_device_train_batch_size"],
    learning_rate=config_toml["learning_rate"],
    weight_decay=config_toml["weight_decay"],
    warmup_ratio=config_toml["warmup_ratio"],
    max_grad_norm=config_toml["max_grad_norm"],
    logging_steps=config_toml["LOG_INTERVAL"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=collator,
    
)

trainer.train()
