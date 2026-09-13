import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler



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
#print(max_cycles[:10])
print(max_cycles.iloc[:10])
print()
df_clean['max_cycles'] = df_clean['unit_id'].map(max_cycles)
print ()
print(df_clean.head())
print()

df_clean["RUL"] = np.minimum((df_clean["max_cycles"] - df_clean["time_cycles"]), 125)
print(df_clean.shape)
print()
print(df_clean.head())
#print(df_clean[df_clean["unit_id"]==1])

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

train_idx, val_idx = next(gss.split(df_clean, groups=df_clean['unit_id']))


df_clean_train = df_clean.iloc[train_idx]
df_clean_val  = df_clean.iloc[val_idx]
print("train count")
print(df_clean_train["unit_id"].nunique())
print()
print("val count")
print(df_clean_val["unit_id"].nunique())
print()
overlap = set(df_clean_train['unit_id']) & set(df_clean_val['unit_id'])
print("Overlap:", overlap)
print()
drop_cols = ['unit_id', 'time_cycles', 'max_cycles', 'op_setting_1', 'op_setting_2', 'RUL']

X_train = df_clean_train.drop(columns=drop_cols)
y_train = df_clean_train['RUL']

X_val = df_clean_val.drop(columns=drop_cols)
y_val = df_clean_val['RUL']
print()

print(X_train.shape, y_train.shape)
print()
print(X_val.shape, y_val.shape)

print()
print()
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

print("Means:")
print(pd.Series(scaler.mean_, index=X_train.columns))

print()

print("Standard deviations:")
print(pd.Series(scaler.scale_, index=X_train.columns))
print()
print()

X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

model = LinearRegression()  
model.fit(X_train_scaled, y_train) 
predictions = model.predict(X_val_scaled)
print()
print("PRECITIONS    ",predictions[:10])
print()
rmse = mean_squared_error(y_val, predictions, squared=False)
print ()
print("RMSE =  ",rmse)
print()

print(model.coef_)
print()
#zipper=zip(X_train.columns, model.coef_

coef_series = pd.Series(model.coef_, index=X_train.columns)

print(coef_series)






