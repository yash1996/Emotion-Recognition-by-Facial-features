import keras
import os
from keras.models import model_from_json
import h5py
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Activation, Input
from keras.optimizers import adam, rmsprop, adadelta
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
json_file = open('model.json','r')
x_test = h5py.File('x_val.hdf5','r')
y_test = h5py.File('y_val.hdf5','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("weights.hdf5")
print("Loaded model from disk")
print(list(x_test.keys()))
print(list(y_test.keys()))
x_test = x_test['x_val'][:]
y_test = y_test['y_val'][:]
y_test = to_categorical(y_test)
print(x_test[0],x_test[1])
print(y_test[0],y_test[1])
#a = model.get_config()
#print("model config is ",a)
model.compile(optimizer='rmsprop', loss='mse')
accuracy = model.evaluate(x_test,y_test)
print('accuracy',accuracy)