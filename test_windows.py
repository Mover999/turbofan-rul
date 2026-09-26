import pandas as pd
import numpy as np




def test_window_maker(df, window_size):
    windows = []
    labels = []
    for unit_id, group in df.groupby('unit_id'):
        #feature_cols = [x for x in df.columns if x not in ['unit_id', 'time_cycles']]
        feature_cols = [x for x in df.columns if x not in ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2']]
        arr = group[feature_cols].to_numpy()
           
        window = arr[-window_size:]
        windows.append (window)
        

    return  np.array(windows)