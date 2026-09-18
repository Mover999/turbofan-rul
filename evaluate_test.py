import joblib 

loaded_linear_model=joblib.load('linear_model.joblib')

loaded_lgb_model=joblib.load('lgb_model.joblib')

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from data_utils import load_cmapps_data

df_clean = load_cmapps_data('data/CMaps/test_FD001.txt')


X_test=df_clean.groupby("unit_id").tail(1)
print(X_test.shape)

drop_cols = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2']

X_test=X_test.drop(columns=drop_cols)

print(X_test.shape)




scaler = StandardScaler()
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)




linear_preds = loaded_linear_model.predict(X_test_scaled)


RUL = pd.read_csv('data/CMaps/RUL_FD001.txt', sep='\s+', engine='python', header=None, names=["RUL"])

rmse = mean_squared_error(
    RUL,
    linear_preds,
    squared=False
)

print("linear  ",rmse)

lgb_preds = loaded_lgb_model.predict(X_test)

rmse = mean_squared_error(
    RUL,
    lgb_preds,
    squared=False
)

print("lgb  ",rmse)
