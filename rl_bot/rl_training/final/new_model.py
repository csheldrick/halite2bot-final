import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from parsing import parse, load_data

seed = 69
np.random.seed(seed)


def normalize_input(input_data):
    # Assert the shape is what we expect
    shape = input_data.shape
    assert len(shape) == 3 and shape[1] == 28 and shape[2] == 11

    m = np.expand_dims(input_data.mean(axis=1), axis=1)
    s = np.expand_dims(input_data.std(axis=1), axis=1)
    return (input_data - m) / (s + 1e-6)


mlp = MLPClassifier(hidden_layer_sizes=(12,6,1), alpha=1, verbose=True)
X, Y = load_data()
data_size = len(X)
train_X, train_Y = X[:int(0.85 * data_size)], Y[:int(0.85 * data_size)]
test_X, test_Y = X[int(0.85 * data_size):], Y[int(0.85 * data_size):]
train_X = np.reshape(train_X, [-1, 11])
test_X = np.reshape(test_X, [-1, 11])
mlp.fit(train_X, train_Y)
mlp.score(test_X, test_Y)
