import glob
import matplotlib.pyplot as plt
import math
import msprime
import numpy as np
import pandas as pd
import PTA
import momi
import os
import time
import datetime
from os import listdir
from os.path import join, isfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from PTA import jmsfs
###########loading data
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object) 
pd.set_option('display.max_columns', 100)
muhi_sigmahi_dat = pd.read_csv("./default_PTA/npops7_nhap1244_hihi_100k_head.csv", sep=" ")
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object) 
muhi_sigmalo_dat = pd.read_csv("./default_PTA/npops7_nhap1244_hilo_100k_head.csv", sep=" ")
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object)
mumid_sigmahi_dat = pd.read_csv("./default_PTA/npops7_nhap1244_midhi_100k_head.csv", sep=" ")
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object)
mumid_sigmalo_dat = pd.read_csv("./default_PTA/npops7_nhap1244_midlo_100k_head.csv", sep=" ")
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
############ test/training data for 2 params
idx = variable_dat.columns.get_loc('0')
variable_params = variable_dat.iloc[:, :idx]
variable_jmsfs = variable_dat.iloc[:, idx:]
variable_p = variable_params[["r_modern_mu", "r_modern_sigma"]]
variable_X_train, variable_X_test, variable_y_train, variable_y_test = train_test_split(variable_jmsfs, variable_p, test_size=0.25)
print("**Training Data Dimensions**")
print(np.shape(variable_X_train))
############ fit RFR for both
rgr = RandomForestRegressor(max_depth=5)
rgr.fit(variable_X_train, variable_y_train)
timestamp = time.time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object) 
variable_y_pred = cross_val_predict(rgr, variable_X_train, variable_y_train, cv=3)
variable_X_train
from sklearn.metrics import *
print("**RFR R2 Values**")
print(r2_score(variable_y_train, variable_y_pred, multioutput='raw_values'))
############ test/training data for rmodmu
idx = variable_dat.columns.get_loc('0')
variable_params = variable_dat.iloc[:, :idx]
variable_jmsfs = variable_dat.iloc[:, idx:]
variable_p = variable_params[["r_modern_mu"]]
variable_X_train, variable_X_test, variable_y_train, variable_y_test = train_test_split(variable_jmsfs, variable_p, test_size=0.25)
print("**Training Data Dimensions**")
print(np.shape(variable_X_train))
############ fit GBR for rmodmu
all_models = {}
common_params = dict(
    learning_rate=0.05,
    n_estimators=200,
    max_depth=2,
    min_samples_leaf=9,
    min_samples_split=9,
)
for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
    all_models["q %1.2f" % alpha] = gbr.fit(variable_X_train, variable_y_train)
gbr_ls = GradientBoostingRegressor(loss="squared_error", **common_params)
all_models["mse"] = gbr_ls.fit(variable_X_train, variable_y_train)

#variable_X_test=variable_X_test.drop(columns='sort_column')
y_pred_train = all_models["mse"].predict(variable_X_train)

print("**GBR Train rmodmu R2 Values**")
print(r2_score(variable_y_train, y_pred_train))

y_pred_test = all_models["mse"].predict(variable_X_test)

print("**GBR Test rmodmu R2 Values**")
print(r2_score(variable_y_test, y_pred_test))

#variable_y_sort=variable_y_test.sort_values(by='r_modern_mu')
# Create a new column in df1 that maps the values in 'Col2' to the values in the dictionary
variable_X_test['sort_column'] = variable_y_test['r_modern_mu']
variable_X_sample = variable_X_test.sample(n=100)
variable_X_sort = variable_X_sample.sort_values('sort_column')
variable_y_sort = variable_X_sort['sort_column'].tolist()
variable_X_sort=variable_X_sort.drop(columns='sort_column')
# Sort df1 based on the new column, then drop the new column


y_pred = all_models["mse"].predict(variable_X_sort)
y_lower = all_models["q 0.05"].predict(variable_X_sort)
y_upper = all_models["q 0.95"].predict(variable_X_sort)
y_med = all_models["q 0.50"].predict(variable_X_sort)


fig = plt.figure(figsize=(10, 10))
plt.plot(variable_y_sort, y_med, "r-", label="Predicted median")
plt.plot(variable_y_sort, y_pred, "r-", label="Predicted mean")
plt.plot(variable_y_sort, y_upper, "k-")
plt.plot(variable_y_sort, y_lower, "k-")
plt.fill_between(
    variable_y_sort, y_lower, y_upper, alpha=0.4, label="Predicted 90% interval"
)
plt.xlabel("true rmodmu")
plt.ylabel("estimated rmodmu")
plt.legend(loc="upper left")
plt.savefig("outputs/quantilereg_rmodmu_100k")


sfs_list = []
path=("/home/br450/ptaPIRE_sims/sfs_empirical_1244_folded")
for filename in listdir(path):
    full_path = join(path, filename)
    sfs_list.append(full_path)
jmsfs = PTA.jmsfs.JointMultiSFS(sfs_list, proportions=True)
y_pred = all_models["mse"].predict(jmsfs.to_dataframe())
y_med = all_models["q 0.50"].predict(jmsfs.to_dataframe())
y_lower = all_models["q 0.05"].predict(jmsfs.to_dataframe())
y_upper = all_models["q 0.95"].predict(jmsfs.to_dataframe())
print("**Estimated rmodmu Value**")
print(y_pred)


############# fit GBR for rmodsigma

idx = variable_dat.columns.get_loc('0')
variable_params = variable_dat.iloc[:, :idx]
variable_jmsfs = variable_dat.iloc[:, idx:]
variable_p = variable_params[["r_modern_sigma"]]
variable_X_train, variable_X_test, variable_y_train, variable_y_test = train_test_split(variable_jmsfs, variable_p, test_size=0.25)
print("**Training Data Dimensions**")
print(np.shape(variable_X_train))

all_models = {}
common_params = dict(
    learning_rate=0.05,
    n_estimators=200,
    max_depth=2,
    min_samples_leaf=9,
    min_samples_split=9,
)
for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
    all_models["q %1.2f" % alpha] = gbr.fit(variable_X_train, variable_y_train)

gbr_ls = GradientBoostingRegressor(loss="squared_error", **common_params)
all_models["mse"] = gbr_ls.fit(variable_X_train, variable_y_train)

#variable_X_test=variable_X_test.drop(columns='sort_column')
y_pred_train = all_models["mse"].predict(variable_X_train)

print("**GBR Train R2 Values**")
print(r2_score(variable_y_train, y_pred_train))

y_pred_test = all_models["mse"].predict(variable_X_test)

print("**GBR Test R2 Values**")
print(r2_score(variable_y_test, y_pred_test))

#variable_y_sort=variable_y_test.sort_values(by='r_modern_sigma')
# Create a new column in df1 that maps the values in 'Col2' to the values in the dictionary
variable_X_test['sort_column'] = variable_y_test['r_modern_sigma']
variable_X_sample = variable_X_test.sample(n=100)
variable_X_sort = variable_X_sample.sort_values('sort_column')
variable_y_sort = variable_X_sort['sort_column'].tolist()
variable_X_sort=variable_X_sort.drop(columns='sort_column')
# Sort df1 based on the new column, then drop the new column


y_pred = all_models["mse"].predict(variable_X_sort)
y_lower = all_models["q 0.05"].predict(variable_X_sort)
y_upper = all_models["q 0.95"].predict(variable_X_sort)
y_med = all_models["q 0.50"].predict(variable_X_sort)


fig = plt.figure(figsize=(10, 10))
plt.plot(variable_y_sort, y_med, "r-", label="Predicted median")
plt.plot(variable_y_sort, y_pred, "r-", label="Predicted mean")
plt.plot(variable_y_sort, y_upper, "k-")
plt.plot(variable_y_sort, y_lower, "k-")
plt.fill_between(
    variable_y_sort, y_lower, y_upper, alpha=0.4, label="Predicted 90% interval"
)
plt.xlabel("true rmodsigma")
plt.ylabel("estimated rmodsigma")
plt.legend(loc="upper left")
plt.savefig("outputs/quantilereg_rmodsigma_100k")

sfs_list = []
path=("/home/br450/ptaPIRE_sims/sfs_empirical_1244_folded")
for filename in listdir(path):
    full_path = join(path, filename)
    sfs_list.append(full_path)
jmsfs = PTA.jmsfs.JointMultiSFS(sfs_list, proportions=True)
y_pred = all_models["mse"].predict(jmsfs.to_dataframe())
y_med = all_models["q 0.50"].predict(jmsfs.to_dataframe())
y_lower = all_models["q 0.05"].predict(jmsfs.to_dataframe())
y_upper = all_models["q 0.95"].predict(jmsfs.to_dataframe())
print("**Estimated rmodsigma Value**")
print(y_pred)
