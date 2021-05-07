# autosubs
Autosubs is a library for Automatic Speech Recognition using a modified version of Listen-Attend-Spell.  The biggest changes are the addition of three bi-directional LSTM layers to the beginning of the network, dropout between all LSTM layers, and the addition of scaling to the dot-product attention mechanism.  The WSJ training process entailed pretraining the decoder for ~15 epochs, and then training hundreds of epochs for between 24-48 hours on 1 Tesla V100.

# Installation
The full conda environment is exported in autosubs_env.yml
You can install this running: `conda env create -f autosubs_env.yml`

# Training and Inference
- Set your configuration in src/config.yaml
    - All hyperparameters and general execution parameters are exposed here for tuning.
- Training is started by running: `python runner.py`
- Inference is started by running: `python inference.py`

## Data
- Datasets belong in the data folder
- For KNNW, these include: 
    - knnw-720p.tar.gz 
    - knnw_en_mono.wav 
    - knnw_en_sub.srt
    - knnw_en.log_spectrogram.npy
    - knnw_en_sub.csv

## Checkpointing
- Models can be trained/evaluated from checkpoint by placing weights in the checkpoints directory, and setting config.yaml's checkpoint path to the weights file.