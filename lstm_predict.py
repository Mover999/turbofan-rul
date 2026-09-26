import joblib
import keras
import math
import numpy as np
import pandas as pd
from data_utils import load_cmapps_data, add_features, window_maker, get_feature_cols
from sklearn.preprocessing import StandardScaler
from data_utils import load_cmapps_data

#RUL = load_cmapps_data('data/CMaps/RUL_FD001.txt')

RUL = pd.read_csv('data/CMaps/RUL_FD001.txt', sep='\s+', engine='python', header=None, names=['RUL'])

print(RUL.head())

loaded_scaler = joblib.load('scaler.joblib')
model=keras.models.load_model('model_lstm.keras')

df=load_cmapps_data('data/CMaps/test_FD001.txt')

#print(df.head(1))
#print("first_list")
#print(df.columns.tolist())

print()
#print("unit_id")
#print(df['unit_id'].values[:10])

df=add_features(df, window_size=10)
#print("after add features")
#Sprint(df.isna().sum().sum())
print()

#print(filtered_df)

#rint(df[df["unit_id"] == 31].shape)

#filtered_df = df[df["unit_id"] == 31]
#print("filtered   ", filtered_df.iloc[19:])


print()
#feature_cols = [x for x in df.columns if x not in ['unit_id', 'time_cycles', 'op_setting_1','op_setting_2']]
feature_cols=get_feature_cols(df)
df_scaled=loaded_scaler.transform(df[feature_cols])
#print(df_scaled[:1])
#rint(np.isnan(df_scaled).any().any())
#print(df.columns.tolist())
#df_scaled = pd.DataFrame(df, columns=df.columns, index=df.index)
#df_scaled = pd.DataFrame(df, columns=df.columns)


df_scaled = pd.DataFrame(df_scaled, columns=feature_cols)

df_scaled['unit_id'] = df['unit_id'].values

#print(df_scaled.columns.tolist())

#===================================================================


def test_window_maker(df, window_size):
    windows = []
    labels = []
    for unit_id, group in df.groupby('unit_id'):
        #feature_cols = [x for x in df.columns if x not in ['unit_id', 'time_cycles']]
        #feature_cols = [x for x in df.columns if x not in ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2']]
        feature_cols=get_feature_cols(df)
        arr = group[feature_cols].to_numpy()
           
        window = arr[-window_size:]
        windows.append (window)
        

    return  np.array(windows)

testing=test_window_maker(df_scaled,window_size=30)

predictions=model.predict(testing)

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(
    RUL,
    predictions,
    squared=False
)

print(math.sqrt(rmse))

