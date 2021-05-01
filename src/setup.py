import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.optim as optim
import torchvision
import yaml
from ray import tune
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from yaml import Loader, load
import Levenshtein
import math
import copy
import os
from torch.nn.functional import log_softmax, gumbel_softmax
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import *

random.seed(123123)
torch.manual_seed(123123)
device = torch.device("cuda:0")

start_time = time.strftime("%Y-%m-%d-%H-%M%S")

# Load Config
cfg_path = "./config.yaml"
with open(cfg_path) as file:
    cfg = yaml.load(file, Loader=Loader)

knnw_audio_path = os.path.abspath(cfg['knnw_audio_path'])
knnw_subtitle_path = os.path.abspath(cfg['knnw_subtitle_path'])
train_path = os.path.expanduser(cfg['train_path'])
train_transcripts_path = os.path.expanduser(cfg['train_transcripts_path'])
val_path = os.path.expanduser(cfg['val_path'])
val_transcripts_path = os.path.expanduser(cfg['val_transcripts_path'])
debug_train_path = os.path.expanduser(cfg['debug_train_path'])
debug_train_transcripts_path = os.path.expanduser(cfg['debug_train_transcripts_path'])
debug_val_path = os.path.expanduser(cfg['debug_val_path'])
debug_val_transcripts_path = os.path.expanduser(cfg['debug_val_transcripts_path'])
test_path = os.path.expanduser(cfg['test_path'])
results_path = os.path.expanduser(cfg['results_path'])
ray_results_dir = os.path.expanduser(cfg['ray_results_dir'])
checkpoints_path = os.path.expanduser(cfg['checkpoints_path'])
plot_path = os.path.expanduser(cfg['plot_path'])
decoder_ckpt_dir = os.path.expanduser(cfg['decoder_ckpt_dir'])
decoder_ckpt_path = os.path.expanduser(cfg['decoder_ckpt_path'])
full_ckpt_path = os.path.expanduser(cfg['full_ckpt_path'])