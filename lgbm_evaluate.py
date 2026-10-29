



from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import math

import joblib
from data_utils import load_cmapps_data, add_features, window_maker, get_feature_cols


from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


from data_utils import load_cmapps_data

df_clean = load_cmapps_data('data/CMaps/test_FD001.txt')
df_clean = add_features(df_clean, window_size=10)


X_test=df_clean.groupby("unit_id").tail(1)
print(X_test.shape)

drop_cols = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2']

X_test=X_test.drop(columns=drop_cols)

print("X_test_shape   ",X_test.shape)

loaded_lgb_model=joblib.load("lgb_model4.joblib")

RUL = pd.read_csv('data/CMaps/RUL_FD001.txt', sep='\s+', engine='python', header=None, names=["RUL"])


lgb_preds = loaded_lgb_model.predict(X_test)

mse2 = mean_squared_error(
    RUL,
    lgb_preds,
    #squared=False
)

print("lgb = ",math.sqrt(mse2))
