import pandas as pd

def load_cmapps_data(filepath):
    col_names = (
    ['unit_id', 'time_cycles'] + 
    ['op_setting_1', 'op_setting_2', 'op_setting_3'] + 
    [f'sensor_{i}' for i in range(1, 22)]
)

    df = pd.read_csv(filepath, sep='\s+', engine='python', header=None, names=col_names)

    df_clean = df.drop(columns=['op_setting_3', 'sensor_1', 'sensor_5',  'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'])
      

    return df_clean