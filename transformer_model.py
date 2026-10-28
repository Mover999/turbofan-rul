import numpy as np
import pandas as pd
import tensorflow
import keras 
import joblib
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from data_utils import load_cmapps_data, add_features, window_maker, get_feature_cols
import sklearn
from sklearn.metrics import mean_squared_error
import math
import statistics

loaded_scaler = joblib.load('scaler.joblib')


RUL = pd.read_csv('data/CMaps/RUL_FD001.txt', sep='\s+', engine='python', header=None, names=['RUL'])


early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)



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

data = np.load("dataset.npz")
loaded_X_train = data['X']
loaded_y_train = data['y']


val=np.load("val.npz")
loaded_X_val=val['X_val']
loaded_y_val=val["y_val"]

positions = np.arange(30) / 29
positions = positions.reshape(1, 30, 1)
positions = np.tile(positions, (loaded_X_train.shape[0], 1, 1))
X_with_pos = np.concatenate([loaded_X_train, positions], axis=-1)


positions = np.arange(30) / 29
positions = positions.reshape(1, 30, 1)
positions = np.tile(positions, (loaded_X_val.shape[0], 1, 1))
X_val_with_pos = np.concatenate([loaded_X_val, positions], axis=-1)

print("Xs")
print(X_with_pos.shape)
print(X_val_with_pos.shape)

from tensorflow.keras import layers
scores=[]
vlosses=[]
train_at_best_list=[]
for i in range (10):
    keras.utils.set_random_seed(i)
    inputs = keras.Input(shape=(30, 46))
    
    x = layers.Dense(64)(inputs)
    x = layers.Dropout(rate=0.3)(x)



    #print (x.shape)

    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = x + attn_output
    x = layers.LayerNormalization()(x)

    ffn_output = layers.Dense(256, activation='relu')(x)
    ffn_output = layers.Dense(64)(ffn_output)

    x = x + ffn_output
    x = layers.LayerNormalization()(x)
    x=x[:, 29, :]
    x=layers.Dense(32, activation='relu')(x)  

    x=layers.Dense(1, activation='linear')(x)
    #print("x.shape")
    #print(x.shape)

    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    history = model.fit(
        X_with_pos,loaded_y_train, 
        validation_data=(X_val_with_pos, loaded_y_val),
        epochs=50, 
        callbacks=[early_stop],
        verbose=1
    )
    print()


    #best = val_hist.index(min(val_hist))       # epoch of the lowest val_loss
    #train_at_best = history.history['loss'][best]   



    vlosses.append(min(history.history['val_loss']))

    
    val_hist = history.history['val_loss']

    best = val_hist.index(min(val_hist))    # epoch of the lowest val_loss
    train_at_best = history.history['loss'][best]   
    print("train_at_best")
    print(train_at_best)

    train_at_best_list.append(train_at_best)

    print()
    #print("vlosses   ", vlosses)
    #print("validation data")
    #print("Validation Loss History:", history.history['val_loss'])
    print()
    model.save('model_transformers.keras')

    print("saved  model")
    saved_model=keras.models.load_model('model_transformers.keras')




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

    print()

    testing=test_window_maker(df_scaled,window_size=30)
    print()

    print("testing shape")
    print(testing.shape)

    positions = np.arange(30) / 29
    positions = positions.reshape(1, 30, 1)
    positions = np.tile(positions, (100, 1, 1))
    testing_with_pos = np.concatenate([testing, positions], axis=-1)

    predictions=saved_model.predict(testing_with_pos)    

    result = mean_squared_error(RUL, predictions)

    print(math.sqrt(result))
    score=math.sqrt(result)
    scores.append(score)

    

print (scores)

m=statistics.mean(scores)
s=statistics.stdev(scores)

print(scores)
print()
print ("mean   ", m)
print ("std  ", s)
print()
vm=statistics.mean(vlosses)
vs=statistics.stdev(vlosses)
print("vmean  ", vm)
print("vstd  ", vs)
print()
train_at_best_mean= statistics.mean(train_at_best_list)
train_at_best_std= statistics.stdev(train_at_best_list)
print()
print()
print("train_at_best_mean " , train_at_best_mean  )
print("train_at_best_std " , train_at_best_std  )





