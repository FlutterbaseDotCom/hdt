from dataclasses import dataclass
import torch
import toml




#WANDB CONFIG
LOAD_SAVED_MODEL    = False
RUN_NUM = 21
WANDB_ID            = "dt_"+str(RUN_NUM)
WNDB_NAME           = "DT_"+str(RUN_NUM)
MODEL_SAVE_NAME     = WNDB_NAME
SAVED_MODEL_VERSION = "latest"



# Define the global device variable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GENERATE DATA CONFIG
NUM_EPISODES = 20
MAX_EPISODE_STEPS = 100

RTG_GAMMA = 1.0


ACTION_PAD_TOKEN_ID = 5 #yakiv.tbd tmp - move to config!!!
ACTION_VOCAB_SIZE = 6 # 5 actions + 1 PAD token

PER_DEVICE_BATCH_SIZE = 16

# TOML-formatted string
config_toml = f"""
PREFIX              = 'DT'
LOG_INTERVAL        = 5
save_steps          = 30
num_train_epochs    = 2
per_device_train_batch_size={PER_DEVICE_BATCH_SIZE}
learning_rate       = 0.0001
weight_decay        = 0.0001
warmup_ratio        = 0.1
max_grad_norm       = 0.25

max_length = 10
max_ep_len = 1000
"""
CONFIG = toml.loads(config_toml)


