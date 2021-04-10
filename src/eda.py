#!/home/gman/anaconda3/bin/python

import seq2seq
from seq2seq import *
import phoneme_list

train_path = os.path.expanduser(cfg['train_path'])
train_labels_path = os.path.expanduser(cfg['train_labels_path'])
val_path = os.path.expanduser(cfg['val_path'])
val_labels_path = os.path.expanduser(cfg['val_labels_path'])

train_data = np.load(train_path, allow_pickle=True)
train_labels = np.load(train_labels_path, allow_pickle=True)
val_data = np.load(val_path, allow_pickle=True)
val_labels = np.load(val_labels_path, allow_pickle=True)


print(f"training utterances : {train_data.shape}")
print(f"train_data[0].shape {train_data[0].shape}")
print(f"train_data[1].shape {train_data[1].shape}")
print(f"labeled utterances: {train_labels.shape}")
print(f"train num labels for utterance 0: {len(train_labels[0])}")
print(f"train num labels for utterance 1: {len(train_labels[1])}")
print(f"val_data shape: {val_data.shape}")
print(f"val_data utterance 0 shape: {val_data[0].shape}")
print(f"val_labels shape: {val_labels.shape}")
print(f"val labels for utterance 0: {val_labels[0]}")


