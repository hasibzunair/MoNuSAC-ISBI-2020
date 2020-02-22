"""
Prediction code for PatchEUnet by the_great_backpropagator


Save final predictions in required format

****Install segmentation_models using this command if the below option fails.****
 - pip install -U --pre segmentation-models --user 


Major requirements:
Python: 3.6
Tensorflow: 2.0.0
Keras: 2.3.1
segmentation_models
OpenCV

"""

# Import libs
import os 

# Install segmentation_models library
os.system("pip install -U --pre segmentation-models --user")

import time
import cv2
from tqdm import tqdm
import numpy as np
import skimage.draw
import random
import keras
import cv2
from glob import glob
import warnings
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage.transform import resize
import efficientnet.tfkeras
from tensorflow.keras.models import load_model

print("All libraries read correctly!!!!")