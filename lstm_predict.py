import joblib
import keras
import math
import numpy as np
import pandas as pd
import math
from data_utils import load_cmapps_data, add_features, window_maker, get_feature_cols
from sklearn.preprocessing import StandardScaler
from data_utils import load_cmapps_data

#RUL = load_cmapps_data('data/CMaps/RUL_FD001.txt')

RUL = pd.read_csv('data/CMaps/RUL_FD001.txt', sep='\s+', engine='python', header=None, names=['RUL'])

#print(RUL.head())

loaded_scaler = joblib.load('scaler.joblib')
model=keras.models.load_model('model_lstm.keras')

#df=load_cmapps_data('data/CMaps/test_FD001.txt')
#df = add_features(df, window_size=10)
#print(df.colu#mns.tolist())

df = load_cmapps_data('data/CMaps/test_FD001.txt')
df = add_features(df, window_size=10)   # <- add this

feature_cols = get_feature_cols(df)
df_scaled = loaded_scaler.transform(df[feature_cols])

'''fit_cols = set(loaded_scaler.feature_names_in_)
current_cols = set(df.columns)

print("In scaler but missing from df:", fit_cols - current_cols)
print("In df but not in scaler:", current_cols - fit_cols)'''

df_scaled = pd.DataFrame(df_scaled, columns=feature_cols)

df_scaled['unit_id'] = df['unit_id'].values

#print(df_scaled.columns.tolist())
print()
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

print("testing shape")
print (testing.shape)
print()
predictions=model.predict(testing)

from sklearn.metrics import mean_squared_error

result= mean_squared_error(
    RUL,
    predictions,
    #squared=False
)
print("RMSE= " , math.sqrt(result))
print()
print()
print(predictions[:10].flatten())
print(RUL['RUL'].values[:10])

