# Operating System and System Tools
import os                          # operating system interfaces
import sys                         # system-specific parameters and functions
import glob                        # working with OS pathnames
import time                        # time processing

# Data Processing and Linear Algebra
import numpy as np                 # linear algebra
import pandas as pd                # data processing, CSV file I/O (e.g. pd.read_csv)

# Data Visualization
import matplotlib.pyplot as plt    # data visualization purposes
import seaborn as sns              # statistical data visualization
sns.set_theme()
from pylab import rcParams         # configuration for matplotlib

# Machine Learning and Metrics
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import (
    cross_val_score,
    TimeSeriesSplit,
    RandomizedSearchCV
)
from sklearn import metrics        # additional metric functions

# StatsModels for Statistical Analysis
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm       # acf, pacf plot
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.tools.eval_measures as em
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from scipy.signal import savgol_filter
from statsmodels.tsa.statespace.sarimax import SARIMAX

# IPython Display Tools
from IPython.display import clear_output, display  # Display functions for Jupyter
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Machine Learning
from sklearn.linear_model import LinearRegression


# TensorFlow and Keras for Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

#External
from copy import deepcopy

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')
