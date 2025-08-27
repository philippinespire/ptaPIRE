import time
import datetime
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print("Starting script - loading packages...")
print(dt_object)
import glob
import matplotlib.pyplot as plt
import math
import msprime
import numpy as np
import pandas as pd
import PTA
import momi
import os
from os import listdir
from os.path import join, isfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PTA import jmsfs
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print("Finished loading")
print(dt_object) 
