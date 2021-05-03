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
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import *
import re
import csv

KNNW_TOTAL_FRAMES = 1370582
KNNW_TOTAL_DURATION = 6396010

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:0")

start_time = time.strftime("%Y-%m-%d-%H-%M%S")

# Load Config
cfg_path = "./config.yaml"
with open(cfg_path) as file:
    cfg = yaml.load(file, Loader=Loader)

knnw_audio_path = os.path.expanduser(cfg['knnw_audio_path'])
knnw_subtitle_path = os.path.expanduser(cfg['knnw_subtitle_path'])
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
wsj_ckpt_path = os.path.expanduser(cfg['wsj_ckpt_path'])
knnw_ckpt_path = os.path.expanduser(cfg['knnw_ckpt_path'])

print("knnw_audio_path", knnw_audio_path)
print("knnw_subtitle_path", knnw_subtitle_path)
print("train_path", train_path)
print("train_transcripts_path", train_transcripts_path)