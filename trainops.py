import tensorflow as tf
from skimage import transform as tr 
from skimage import io 
import pandas as pd 
import numpy as np 

#fetch data 
def datFetch(dat, batch_size, iteration = 0, offset = 0, aug_chance = 0.3):
    
#augment data