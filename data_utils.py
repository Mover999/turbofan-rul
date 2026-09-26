import pandas as pd
import numpy as np

def load_cmapps_data(filepath):
    col_names = (
    ['unit_id', 'time_cycles'] + 
    ['op_setting_1', 'op_setting_2', 'op_setting_3'] + 
    [f'sensor_{i}' for i in range(1, 22)]
)

    df = pd.read_csv(filepath, sep='\s+', engine='python', header=None, names=col_names)

    df_clean = df.drop(columns=['op_setting_3', 'sensor_1', 'sensor_5',  'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'])
    
    return df_clean

def add_features(df, window_size):

    sensor_cols = [x for x in df.columns if x.startswith('sensor_')]
  
    roll_mean_cols = [x + "_roll_mean" for x in sensor_cols]

    df[roll_mean_cols] = df.groupby('unit_id')[sensor_cols].transform(lambda x: x.rolling(window_size).mean())

    sensor_x_rate_change = [x + "_rate_change" for x in sensor_cols]


    df[sensor_x_rate_change] = df.groupby('unit_id')[sensor_cols].transform(lambda x: x.diff(periods=window_size))
    
    #df = df.groupby('unit_id').bfill()

    fill_cols = roll_mean_cols + sensor_x_rate_change
    df[fill_cols] = df.groupby('unit_id')[fill_cols].transform(lambda x: x.bfill())
    return df
 


  



def window_maker(df, window_size):
    windows = []
    labels = []
    for unit_id, group in df.groupby('unit_id'):
        #feature_cols = [x for x in df.columns if x not in ['unit_id', 'time_cycles', 'RUL']]

        feature_cols = [x for x in df.columns if x not in ['unit_id', 'time_cycles', 'RUL', 'op_setting_1', 'op_setting_2']]
        arr = group[feature_cols].to_numpy()
        gp=group['RUL'].to_numpy()

        num_windows = len(arr) - window_size + 1
     
        for i in range(num_windows):
            window = arr[i : i + window_size]
            windows.append (window)
            rul=gp[i + window_size - 1]
            labels.append(rul)

    return  np.array(windows), np.array(labels)
            

            #numbers = np.array(range(1, 31))


def get_feature_cols(df):
    return [x for x in df.columns if x not in ['unit_id', 'time_cycles', 'RUL', 'op_setting_1', 'op_setting_2']]

