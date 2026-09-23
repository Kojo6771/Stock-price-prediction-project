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

#Filtering the dataset for selected companies
companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'EBAY', 'CSCO', 'IBM']

#Plotting the closing and opening prices for the selected companies
plt.figure(figsize=(15,8))
for index, company in enumerate(companies, 1):
    plt.subplot(3,3,index)
    c =data[data['Name'] == company]
    plt.plot(c['date'], c['close'], c='r', label="close", marker="+")
    plt.plot(c['date'], c['open'], c="g", label="open", marker="^")
    plt.title(company)
    plt.legend()
    plt.tight_layout()

#Plotting the trading volume for the selected companies
plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data['Name'] == company]
    plt.plot(c['date'], c['volume'], c='purple', marker='*')
    plt.title(f"{company} Volume")
    plt.tight_layout()

