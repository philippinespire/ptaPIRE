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
pd.set_option('display.max_columns', 100)
muhi_sigmahi_dat = pd.read_csv("./default_PTA/npops7_nhap1244_hihi_500k_head.csv", sep=" ")
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object) 
muhi_sigmalo_dat = pd.read_csv("./default_PTA/npops7_nhap1244_hilo_500k_head.csv", sep=" ")
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object)
mumid_sigmahi_dat = pd.read_csv("./default_PTA/npops7_nhap1244_midhi_500k_head.csv", sep=" ")
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object)
mumid_sigmalo_dat = pd.read_csv("./default_PTA/npops7_nhap1244_midlo_500k_head.csv", sep=" ")
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object)
variable_dat=pd.concat([muhi_sigmahi_dat,muhi_sigmalo_dat,mumid_sigmahi_dat,mumid_sigmalo_dat])
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object)
lst=[muhi_sigmahi_dat,muhi_sigmalo_dat,mumid_sigmahi_dat,mumid_sigmalo_dat]
del lst
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print("Finished loading data and freeing up memory")
print(dt_object)
idx = variable_dat.columns.get_loc('0')
variable_params = variable_dat.iloc[:, :idx]
variable_jmsfs = variable_dat.iloc[:, idx:]
variable_p = variable_params[["r_modern_mu", "r_modern_sigma"]]
variable_X_train, variable_X_test, variable_y_train, variable_y_test = train_test_split(variable_jmsfs, variable_p, test_size=0.25)
print("**Training Data Dimensions**")
print(np.shape(variable_X_train))
rgr = RandomForestRegressor(max_depth=5)
rgr.fit(variable_X_train, variable_y_train)
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object) 
variable_y_pred = cross_val_predict(rgr, variable_X_train, variable_y_train, cv=3)
variable_X_train
from sklearn.metrics import *
print("**R2 Values**")
print(r2_score(variable_y_train, variable_y_pred, multioutput='raw_values'))
t = "r_modern_mu"
plt.scatter(variable_y_train[t], variable_y_pred[:, 0], alpha=0.01)
plt.savefig("outputs/npops7_nhap1244_mumidhi_sigmalohi_muplot_public_300k_nocol")
plt.close()
t = "r_modern_sigma"
plt.scatter(variable_y_train[t], variable_y_pred[:, 1], alpha=0.01)
plt.savefig("outputs/npops7_nhap1244_mumidhi_sigmalohi_sigmaplot_public_300k_nocol")
sfs_list = []
path=("/home/br450/ptaPIRE_sims/sfs_empirical_1244_folded")
for filename in listdir(path):
    full_path = join(path, filename)
    sfs_list.append(full_path)
jmsfs = PTA.jmsfs.JointMultiSFS(sfs_list, proportions=True)
y_pred = rgr.predict(jmsfs.to_dataframe())
print("**Estimated Values**")
print(y_pred)
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object)
