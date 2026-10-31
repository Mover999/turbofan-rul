
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras import layers

from sklearn.metrics import mean_squared_error
import math
import statistics

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

data = np.load("dataset.npz")
loaded_X_train = data['X']
loaded_y_train = data['y']

val=np.load("val.npz")
loaded_X_val=val['X_val']
loaded_y_val=val["y_val"]

scores=[]

for i in range (10):
    keras.utils.set_random_seed(i)





    model =Sequential([
        LSTM(units=50, activation='tanh', input_shape=(30, 45)),
        layers.Dense(32, activation='relu'),   
        layers.Dense(1, activation='linear')  
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])



    history = model.fit(
        loaded_X_train,loaded_y_train, 
        validation_data=(loaded_X_val, loaded_y_val),
        epochs=50, 
        callbacks=[early_stop],
        verbose=1
    )


    model.save('model_lstm.keras')

    #==========================================================
    #===============================================================



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

    print(result)
    score=math.sqrt(result)
    print()
    print()
    scores.append(score)

m=statistics.mean(scores)
s=statistics.stdev(scores)

print(scores)
print()
print ("mean   ", m)
print ("std  ",s)