from keras.optimizers import Adam
from keras.layers import (
    Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate, BatchNormalization,
    Dropout, Lambda, Dense, Activation, Flatten, SpatialDropout2D, UpSampling2D,Conv3D,Conv3DTranspose,MaxPooling3D
)
from keras.metrics import MeanIoU,AUC,Precision,Recall,Accuracy
from keras.models import Model, Sequential
from keras.losses import CategoricalCrossentropy
from keras.utils import Sequence
from tensorflow.keras import regularizers
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from models import convglu_nestedunet,data_loader



model = convglu_nestedunet.nestedunet_convglu(128, 128, 128, 3, 4)

train_dir="DATASET/"
val_dir="DATASET/"

train_data_loader = data_loader.CustomDataLoader(train_dir, batch_size=1, mode='train', shuffle=False)
val_data_loader = data_loader.CustomDataLoader(val_dir, batch_size=1, mode='val', shuffle=False)

weights_folder="weightsfolder"
filepath=weights_folder+"/3dconvglu_nestedunet.hdf5"
model_checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    save_weights_only=True,
    monitor='loss',
    mode='max',
    save_best_only=True)
history=model.fit(train_data_loader,batch_size=1,workers=1,callbacks=[model_checkpoint_callback], epochs=10, validation_data=val_data_loader)
