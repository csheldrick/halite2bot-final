#import torch
#from torch.autograd import Variable
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#import platform
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Reshape, LSTM, Dense, Conv1D, Conv2D, MaxPooling2D, TimeDistributed, Lambda
from keras.models import Model

class KerasNet:

    def __init__(self):
        i = Input(shape=(None,None,7))
        #d = Reshape((-1,7))(i)
        d = TimeDistributed(LSTM(32, activation='sigmoid', return_sequences=True))(i)
        d = TimeDistributed(LSTM(32, activation='sigmoid', return_sequences=True))(d)
        d = TimeDistributed(LSTM(32, activation='sigmoid', return_sequences=True))(d)
        d = TimeDistributed(LSTM(32, activation='sigmoid', return_sequences=True))(d)
        #d = LSTM(32, activation='sigmoid', return_sequences=True)(d)
        #d = LSTM(32, activation='sigmoid', return_sequences=True)(d)
        #d = LSTM(3, activation='sigmoid', return_sequences=True)(d)
        d = TimeDistributed(Dense(32, activation='relu'))(d)
        d = TimeDistributed(Dense(3, activation='sigmoid'))(d)
        self.model = Model(i, d)
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])
        self.model.summary()

    def my_train(self, inputs, labels, epochs=10):
        if type(inputs) == list:
            inputs = np.array(inputs)
            labels = np.array(labels)
        print(inputs.shape)
        for epoch in range(1,epochs+1):  # loop over the dataset multiple times
            print("Epoch:", epoch, "out of:", epochs)
            history = self.model.fit(inputs, labels, batch_size=32, epochs=epochs, verbose=1)
            #print("loss {}".format(history[0]))
        loss = self.model.evaluate(np.array([inputs[0]]), np.array([labels[0]]))
        print(loss)
