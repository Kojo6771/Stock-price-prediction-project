#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import os
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

#Loading the dataset
data =pd.read_csv('all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')
print(data.shape)
print(data.sample(7))

#Exploring the dataset
data.info()

data['date'] = pd.to_datetime(data['date'])
data.info()


