
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras import layers

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

model =Sequential([
    LSTM(units=50, activation='tanh', input_shape=(30, 45)),
    layers.Dense(32, activation='relu'),   
    layers.Dense(1, activation='linear')  
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.save('model_lstm.keras')

history = model.fit(
    loaded_X_train,loaded_y_train, 
    validation_data=(loaded_X_val, loaded_y_val),
    epochs=50, 
    callbacks=[early_stop],
    verbose=1
)
