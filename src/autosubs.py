import os
import time
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import csv
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime as dt
from torch.nn.utils.rnn import *
import yaml
from yaml import load, Loader
from KnnwAudioDataset import KnnwAudioDataset
import torchaudio

KNNW_TOTAL_FRAMES = 1370582
KNNW_TOTAL_DURATION = 6396010

cfg_path = "./config.yaml"
with open(cfg_path) as file:
    cfg = yaml.load(file, Loader=Loader)


def process_train_data(data):
    """
    Performs SpecAugment data preprocessing
    Returns:
        Tensor  The processed audio data
    """
    training_audio_transforms = torch.nn.Sequential(
        # TODO: Add in time stretching if needed. SpecAugment papers indicates 
        #       it is less effective than the other two techniques which is the
        #       reason it was left out.
        # Params indicate maximum possible length (in seq indices) of the masks
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    )
    proc_data = training_audio_transforms(torch.transpose(data, 2, 1))
    return torch.transpose(proc_data, 2, 1)

def pad_collate_fn(data, mode='train'):
    data.sort(reverse=True, key=lambda x: x[0].shape[1])
    
    # Audio data is of the form (batch, seq_len, feature_len)
    audio, subtitles, audio_lengths = zip(*data)
    audio = [torch.from_numpy(item).float() for item in audio]
    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
    if mode == 'train':
        audio = process_train_data(audio)
    
    return audio, subtitles, audio_lengths

def create_train_only_data():
    train_dataset = KnnwAudioDataset(cfg['train_path'], cfg['train_labels'], KNNW_TOTAL_FRAMES, KNNW_TOTAL_DURATION)
    return train_dataset

def split_into_val_train():
    """
    Splits data into train and val data using SubsetRandomSampler. 
    Returns:
        (DataLoader, DataLoader)    Train and validation loaders
    """
    dataset = create_train_only_data()
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(cfg['validation_split'] * dataset_size))
    if cfg['shuffle_dataset_split'] :
        np.random.seed(cfg['random_seed_split'])
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset = dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], 
                              pin_memory=cfg['pin_memory'], drop_last = not cfg['DEBUG'], sampler=train_sampler, 
                              collate_fn=pad_collate_fn)
    val_loader = DataLoader(dataset = dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], 
                            pin_memory=cfg['pin_memory'], drop_last = not cfg['DEBUG'], sampler=valid_sampler, 
                            collate_fn=lambda x: pad_collate_fn(x, 'valid'))

    return train_loader, val_loader


train_loader, val_loader = split_into_val_train()
dataset = train_loader.dataset
print("Number of subtitles in train", len(train_loader))
print("Number of subtitles in val", len(val_loader))
print("Train Data example:", next(iter(dataset)), sep="\n")
