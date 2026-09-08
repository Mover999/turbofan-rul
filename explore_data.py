import pandas as pd
import numpy as np

col_names = (
    ['unit_id', 'time_cycles'] + 
    ['op_setting_1', 'op_setting_2', 'op_setting_3'] + 
    [f'sensor_{i}' for i in range(1, 22)]
)


df = pd.read_csv('data/CMaps/train_FD001.txt', sep='\s+', engine='python', header=None, names=col_names)

print(df.shape)
print("original")
print()
print(df.head())
print()

print("unique  ", df['unit_id'].nunique())
print()
#print(print(df.describe()))

summary=df.describe()

stdx=summary.loc['std']
#print(stdx)
print()
df_clean = df.drop(columns=['op_setting_3', 'sensor_1', 'sensor_5',  'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'])
print(df_clean.shape)
print (df_clean.head())
print()
print("tail")
print()
print(df_clean.tail())
print()
max_cycles = df_clean.groupby('unit_id')['time_cycles'].max()
print("length   ",len(max_cycles))
print("max_cycles")
print()
print(max_cycles[:10])
print()
df_clean['max_cycles'] = df_clean['unit_id'].map(max_cycles)
print ()
print(df_clean.head())
print()

df_clean["RUL"] = np.minimum((df_clean["max_cycles"] - df_clean["time_cycles"]), 125)
print(df_clean.shape)
print()
print(df_clean.head())
print(df_clean[df_clean["unit_id"]==1])

