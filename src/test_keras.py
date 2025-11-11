import tensorflow as tf
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Conv2D
from keras.layers import LeakyReLU, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.optimizers import Adam
from keras.models import load_model
from omegaconf import DictConfig, OmegaConf
import glob
from pathlib import Path
attacks_dict = {'Flooding': 'test_flooding',
        'Suppress': 'test_suppress',
        'Plateau': 'test_plateau',
        'Continuous': 'test_continuous',
        'Playback': 'test_playback'}

df = pd.read_csv("data/results/syncan/baseline_cm.csv")
print(df.describe())

for file_name in attacks_dict.keys():
    df = df[df['Model'] == 'CANet']
    print(df[f'{file_name}_fpr'].values[0])














