from scipy.misc import imsave
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, TimeDistributed, BatchNormalization, SeparableConv2D, Conv2DTranspose
from keras.layers import PReLU, LeakyReLU, ThresholdedReLU, Masking, Dropout, Lambda, multiply, add, subtract, concatenate, AlphaDropout, LSTM, ConvLSTM2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal, Ones, Constant, RandomUniform, normal, lecun_normal, RandomNormal, Orthogonal
from keras.regularizers import l2, l1
from keras.constraints import non_neg, max_norm, unit_norm, min_max_norm
from TrainingBot import NUM_FEATURES, NUM_OUTPUT_FEATURES

# Ship features = [:,:,:4], Planet Features = [:,:,4:]
def split(x, half='first'):
    if half == 'first':
        return x[:,:,:,:3]
    else:
        return x[:,:,:,3:]
    #ship = Lambda(lambda x: split(x))(i)
    #planet = Lambda(lambda x: split(x, "last"))(i)

# Softplus = f(x) = ln(1+e^x)              range: (0, inf)
# Relu = f(x) = 0 for x < 0 x for x >= 0   range: (0, inf)
# Softsign = f(x) = x / ( 1 + abs(x))      range: (-1, 1)
# Tanh = f(x) = (2 / (1 + e^-2x)) - 1      range: (-1, 1)
# Sigmoid = f(x) = 1 / ( 1 + e^-x)         range: (0, 1)
# Binary = f(x) = 0 for x < 0 1 for x >= 0 range: {0 or 1}
def LeakyConv(f, k, pad="same", i=None):
    x = Conv2D(f, k, padding=pad, kernel_initializer='he_normal')(i)
    x = LeakyReLU()(x)
    return x

class KerasNet:
    def __init__(self):
        # "Full" net
        i = Input(shape=(120,80,NUM_FEATURES))
        # "Angle" net sees (3x3x28)x4 -> 3x3x1
        d = LeakyConv(24, 8, pad='same', i=i)
        d = LeakyConv(12, 4, pad='same', i=d)
        d = LeakyConv(6, 2, pad='same', i=d)
        d = Conv2D(3, 1, padding='same')(d) #, kernel_initializer="ones")(d)
        d = Activation('sigmoid')(d) #LeakyReLU()(d)
        # "Speed" net sees (3x3x48)x4 -> 3x3x1
        #x = LeakyConv(20, 3, pad='same', i=i)
        #x = LeakyConv(20, 3, pad='same', i=x)
        #x = LeakyConv(20, 3, pad='same', i=x)
        #x = Conv2D(1, 3, padding='same')(x) #, kernel_initializer="ones")(x)
        #x = Activation('relu')(x) #LeakyReLU()(d)
        # "Dock" net sees (3x3x24)x4 -> 3x3x1
        #z = LeakyConv(24, 3, pad='same', i=i)
        #z = LeakyConv(24, 3, pad='same', i=z)
        #z = LeakyConv(24, 3, pad='same', i=z)
        #z = Conv2D(1, 3, padding='same')(z) #, kernel_initializer='ones')(z)
        #z = Activation('relu')(z) #LeakyReLU()(z)
        # Concatenate Angle, Speed, Dock into output
        #d = Activation('relu')(d)
        #x = Activation('relu')(x)
        #z = Activation('sigmoid')(z)
        #d = concatenate([d,x,z])
        self.model = Model(i, d)
        a = Adam(lr=1e-4)
        self.model.compile(loss='mse', optimizer=a)
        self.model.summary()


    def my_train(self, inputs, labels, epochs=10):
        inputs = np.array(inputs)
        labels = np.array(labels)
        history = self.model.fit(inputs, labels, batch_size=32, epochs=epochs, verbose=1)
        print("loss {}".format(history.history))
        loss = self.model.evaluate(inputs, labels)
        print(loss)
        #print(self.model.predict(inputs[0]))
