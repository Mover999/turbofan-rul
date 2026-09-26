import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import joblib

import lightgbm as lgb

import shap
import matplotlib.pyplot as plt
from data_utils import load_cmapps_data, add_features, window_maker, get_feature_cols



df_clean = load_cmapps_data('data/CMaps/train_FD001.txt')
df_clean = add_features(df_clean, window_size=10)

print("unique  ", df_clean['unit_id'].nunique())
print()
#print(print(df.describe()))

summary=df_clean.describe()

stdx=summary.loc['std']
#print(stdx)
print("shape")
print()
print(df_clean.shape)
print(df_clean.head())

print()
max_cycles = df_clean.groupby('unit_id')['time_cycles'].max()
print("length   ",len(max_cycles))
print("max_cycles")
print()
#print(max_cycles[:10])
print(max_cycles.iloc[:10])
print()
df_clean['max_cycles'] = df_clean['unit_id'].map(max_cycles)
print ()
print(df_clean.head())

df_clean["RUL"] = np.minimum((df_clean["max_cycles"] - df_clean["time_cycles"]), 125)

df_clean = df_clean.drop(columns=['max_cycles'])
print("df clean shape   ",df_clean.shape)

print()
#print(df_clean.head())
#print(df_clean[df_clean["unit_id"]==1])

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

train_idx, val_idx = next(gss.split(df_clean, groups=df_clean['unit_id']))


df_clean_train = df_clean.iloc[train_idx]
df_clean_val  = df_clean.iloc[val_idx]
print("train count")
print(df_clean_train["unit_id"].nunique())
print("shape")
print(df_clean_train.shape)
print()
print("val count")
print(df_clean_val["unit_id"].nunique())
print()
overlap = set(df_clean_train['unit_id']) & set(df_clean_val['unit_id'])
print("Overlap:", overlap)
print()
drop_cols = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'RUL']

X_train = df_clean_train.drop(columns=drop_cols)
y_train = df_clean_train['RUL']

X_val = df_clean_val.drop(columns=drop_cols)
y_val = df_clean_val['RUL']
print()

print(X_train.shape, y_train.shape)
print()
print(X_val.shape, y_val.shape)

#feature_cols = [x for x in X_train.columns if x.startswith('sensor_')]

feature_cols=get_feature_cols(df_clean_train)

scaler = StandardScaler()
df_clean_train[feature_cols] = scaler.fit_transform(df_clean_train[feature_cols])
df_clean_val[feature_cols] = scaler.transform(df_clean_val[feature_cols])

X_train_windows, y_train_windows = window_maker(df_clean_train, window_size=30)

window_X=X_train_windows
label_y=y_train_windows

np.savez('dataset.npz', X=window_X, y=label_y)




#print("X_train_windows    ", len(X_train_windows))

print(X_train_windows.shape)

print()
print()

X_val_windows, y_val_windows = window_maker(df_clean_val, window_size=30)

X_val=X_val_windows
y_val=y_val_windows

np.savez('val.npz', X_val=X_val_windows, y_val=y_val_windows)

print("X_val windows    ", X_val_windows.shape)

#window_maker(df_clean,window_size=30)






