import pandas as pd
import numpy as np
import math

from sklearn.model_selection import GroupShuffleSplit 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

import lightgbm as lgb

import shap
from data_utils import load_cmapps_data, add_features, window_maker, get_feature_cols

col_names = (
    ['unit_id', 'time_cycles'] + 
    ['op_setting_1', 'op_setting_2', 'op_setting_3'] + 
    [f'sensor_{i}' for i in range(1, 22)]
)


df = pd.read_csv('data/CMaps/train_FD001.txt', sep='\s+', engine='python', header=None, names=col_names)

print(df.shape)
#rint("original")


print("unique  ", df['unit_id'].nunique())

#print(print(df.describe()))

summary=df.describe()

stdx=summary.loc['std']
#print(stdx)


df_clean = df.drop(columns=['op_setting_3', 'sensor_1', 'sensor_5',  'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'])
df_clean=add_features(df_clean, window_size=10)
#df = add_features(df, window_size=10) 

max_cycles = df_clean.groupby('unit_id')['time_cycles'].max()
print("length   ",len(max_cycles))
print("max_cycles")
#print(max_cycles[:10])
#print(max_cycles.iloc[:10])

df_clean['max_cycles'] = df_clean['unit_id'].map(max_cycles)

df_clean["RUL"] = np.minimum((df_clean["max_cycles"] - df_clean["time_cycles"]), 125)
print(df_clean.shape)
#rint(df_clean.head())
#print(df_clean[df_clean["unit_id"]==1])

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

train_idx, val_idx = next(gss.split(df_clean, groups=df_clean['unit_id']))


df_clean_train = df_clean.iloc[train_idx]
df_clean_val  = df_clean.iloc[val_idx]
print("train count")
print(df_clean_train["unit_id"].nunique())

print("val count")
print(df_clean_val["unit_id"].nunique())

overlap = set(df_clean_train['unit_id']) & set(df_clean_val['unit_id'])
print("Overlap:", overlap)

drop_cols = ['unit_id', 'time_cycles', 'max_cycles', 'op_setting_1', 'op_setting_2', 'RUL']

X_train = df_clean_train.drop(columns=drop_cols)
y_train = df_clean_train['RUL']

X_val = df_clean_val.drop(columns=drop_cols)
y_val = df_clean_val['RUL']

print("X_train_shape, y_train_shape")
print(X_train.shape, y_train.shape)

print(X_val.shape, y_val.shape)

#scaler = StandardScaler()

#X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

#int("Means:")
#print(pd.Series(scaler.mean_, index=X_train.columns))



#rint("Standard deviations:")
#print(pd.Series(scaler.scale_, index=X_train.columns))

#===============================================

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31
    
} 

train_data = lgb.Dataset(X_train, label=y_train)


#num_feature = train_data.num_feature()
#print("num feat  ", num_feature) 
#rint("train_datashape =  ", train_data.shape) 

eval_data = lgb.Dataset(X_val, label=y_val, reference=train_data)


lgb_model = lgb.train(
    params,
    train_set=train_data,
    num_boost_round=100,
    valid_sets=[train_data, eval_data],
    valid_names=['train', 'valid'],
)
joblib.dump(lgb_model, "lgb_model4.joblib")

loaded_model=joblib.load("lgb_model4.joblib")
predictions = loaded_model.predict(X_val)

from sklearn.metrics import mean_squared_error

result = mean_squared_error(
    y_val,
    predictions,
    #squared=False
)

print("LightGBM RMSE =", math.sqrt(result))



