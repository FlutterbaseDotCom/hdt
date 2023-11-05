from dataclasses import dataclass
import torch
import toml


#WANDB CONFIG
LOAD_SAVED_MODEL    = False
RUN_NUM = 20
WANDB_ID            = "dt_"+str(RUN_NUM)
WNDB_NAME           = "DT_"+str(RUN_NUM)
MODEL_SAVE_NAME     = WNDB_NAME
SAVED_MODEL_VERSION = "latest"



# Define the global device variable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ACTION_PAD_TOKEN_ID = 5 #yakiv.tbd tmp - move to config!!!
ACTION_VOCAB_SIZE = 6 # 5 actions + 1 PAD token

# TOML-formatted string
config_toml = """
PREFIX              = 'DT'
LOG_INTERVAL        = 5
save_steps          = 50
num_train_epochs    = 1
per_device_train_batch_size=64
learning_rate       = 0.0001
weight_decay        = 0.0001
warmup_ratio        = 0.1
max_grad_norm       = 0.25

max_length = 20
max_ep_len = 1000
"""
CONFIG = toml.loads(config_toml)


