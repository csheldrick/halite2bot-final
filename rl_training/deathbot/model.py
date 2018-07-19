import sys, os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras.backend as K
import tensorflow as tf 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, Reshape, InputLayer, Lambda, LSTM, Masking, \
    TimeDistributed, Input, Embedding, Layer, Activation, GlobalAveragePooling1D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Callback, EarlyStopping
from keras.layers.normalization import BatchNormalization

sys.stderr = stderr
import numpy as np

try:
    from .common import PLANET_MAX_NUM, PER_PLANET_FEATURES
except:
    from common import PLANET_MAX_NUM, PER_PLANET_FEATURES

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def get_input_shape(batch=False):
    return (None, PLANET_MAX_NUM, PER_PLANET_FEATURES) if not batch else [None, PLANET_MAX_NUM, PER_PLANET_FEATURES]


def make_layer_str(layer):
    d = layer.__dict__
    name = layer.name.split("_")[0]
    if hasattr(layer, "units"): units = layer.units
    elif hasattr(layer, "filters"): filters = layer.filters
    if hasattr(layer, "activation"): activation = layer.activation.__name__
    if hasattr(layer, "return_sequences"): return_sequences = layer.return_sequences
    if hasattr(layer, "unroll"): unroll = layer.unroll
    if hasattr(layer, "stateful"): stateful = layer.stateful
    if "dense" in name:
        name = name[0].upper()+name[1:]
    elif "lstm" in name:
        name = name.upper()
    elif "conv" in name:
        name = name[0].upper()+name[1:-1]+name[-1].upper()
    return "{}({}, activation={})".format(name, units, activation)


def make_exp_params(exp, layers):
    if exp is None: return
    for i, layer in enumerate(layers):
        # _layer = make_layer_str(layer)
        try:
            config = layer.get_config()
            for k,v in config.items():
                exp.param("Layer {}-{}".format(i+1, k), str(v))
        except:
            continue


def evo_model(exp=None):
    activations = ['relu', 'sigmoid']
    optimizers = ['adam', 'rmsprop', 'adagrad', 'adadelta']
    dropouts = [i for i in range(11)]
    pools = list(range(3))
    batch_norms = [False, True]
    conv0 = {
        "type": Conv1D,
        "filters": 16,
        "active": True,
        "batch_norm": batch_norms[1],
        "activation": activations[1],
        "dropout": dropouts[2],
        "max_pooling": pools[2]
    }
    conv1 = {
        "type": Conv1D,
        "filters": 16,
        "active": False,
        "batch_norm": batch_norms[0],
        "activation": activations[1],
        "dropout": dropouts[5],
        "max_pooling": pools[1]
    }
    dense0 = {
        "type": Dense,
        "active": True,
        "nodes": 256,
        "batch_norm": batch_norms[1],
        "activation": activations[1],
        "dropout": dropouts[1]
    }
    dense1 = {
        "type": Dense,
        "active": False,
        "nodes": 128,
        "batch_norm": batch_norms[1],
        "activation": activations[0],
        "dropout": dropouts[6]
    }
    dense2 = {
        "type": Dense,
        "active": False,
        "nodes": 16,
        "batch_norm": batch_norms[1],
        "activation": activations[0],
        "dropout": dropouts[9]
    }

    conv_layers = [conv0, conv1]
    dense_layers = [dense0, dense1, dense2]
    input_layer = InputLayer(input_shape=(28, 11))
    output_layer = Dense(28, activation='softmax')
    all_layers = [input_layer]+conv_layers+dense_layers+[output_layer]
    make_exp_params(exp, all_layers)
    model = Sequential()
    model.add(input_layer)
    for layer in conv_layers:
        model.add(layer["type"](layer["filters"], (3,), padding="same"))
        if layer["batch_norm"]: model.add(BatchNormalization())
        model.add(Activation(layer["activation"]))
        model.add(Dropout(float(layer["dropout"] / 20.0)))
        model.add(MaxPooling1D(layer["max_pooling"], padding="same"))
    model.add(Flatten())
    for layer in dense_layers:
        model.add(layer["type"](layer["nodes"]))
        if layer["batch_norm"]: model.add(BatchNormalization())
        model.add(Activation(layer["activation"]))
        model.add(Dropout(float(layer["dropout"] / 20.0)))
    model.add(output_layer)
    return model


def conv_model(exp=None):
    # conditional[1], conv1d_1[1], batch_size[5], conv1d[2], optimizer[3], dense[1]
    # Conv1D(100, kernel_size=(11,), strides=1), PReLU(), MaxPooling1D(2, strides=2), 
    # Conv1D(28, kernel_size=(4,)), PReLU(), MaxPooling1D(2), Flatten(),
    # Dense(256), PReLU(), 
    pool_size = 2
    conv_depth = 100
    kernel1 = (11,)
    conv2_depth = 28
    kernel2 = (4,)
    dense_size = 256
    num_classes = PLANET_MAX_NUM
    num_features = PER_PLANET_FEATURES
    layers = [
        InputLayer(input_shape=(num_classes, num_features)),
        BatchNormalization(),
        Conv1D(conv_depth, kernel1, strides=1),
        PReLU(),
        MaxPooling1D(pool_size=pool_size, strides=2),
        Conv1D(conv2_depth, kernel2),
        PReLU(),
        MaxPooling1D(pool_size=pool_size),
        Dense(dense_size),
        PReLU(),
        Flatten(),
        Dense(num_classes, activation='softmax')
    ]
    make_exp_params(exp, layers)
    return Sequential(layers)


def conv_model2(exp=None):
    pool_size = 2
    drop_prob1 = 0.5
    drop_prob2 = 0.2
    hidden_size = 512
    num_classes = PLANET_MAX_NUM
    num_features = PER_PLANET_FEATURES
    layers = [
        InputLayer(input_shape=(28, 11)),
        BatchNormalization(),
        Conv1D(128, kernel_size=(11,), strides=1),
        PReLU(),
        MaxPooling1D(pool_size=pool_size, strides=2),
        Conv1D(128, (1,)),
        PReLU(),
        MaxPooling1D(pool_size=pool_size),
        Flatten(),
        Dense(1024),
        PReLU(),
        # Dense(1),
        # PReLU(),
        # Dense(256),
        # PReLU(),
        # Dense(128),
        # PReLU(),
        # Dense(64),
        # PReLU(),
        Dense(num_classes, activation='softmax')
    ]
    make_exp_params(exp, layers)
    return Sequential(layers)


def conv_model3(exp=None):
    pool_size = 2
    drop_prob1 = 0.5
    drop_prob2 = 0.2
    hidden_size = 512
    num_classes = PLANET_MAX_NUM
    num_features = PER_PLANET_FEATURES
    layers = [
        InputLayer(input_shape=(28, 11)),
        BatchNormalization(),
        Conv1D(28, kernel_size=(11,), strides=1),
        PReLU(),
        MaxPooling1D(pool_size=pool_size, strides=2),
        Conv1D(28, (1,)),
        PReLU(),
        MaxPooling1D(pool_size=pool_size),
        Flatten(),
        Dense(hidden_size),
        PReLU(),
        Dense(hidden_size),
        PReLU(),

        Dense(num_classes, activation='softmax')
    ]
    make_exp_params(exp, layers)
    return Sequential(layers)


def conv_model4(exp=None):
    pool_size = 2
    drop_prob1 = 0.5
    drop_prob2 = 0.2
    hidden_size = 512
    num_classes = PLANET_MAX_NUM
    num_features = PER_PLANET_FEATURES
    layers = [
        InputLayer(input_shape=(28, 11)),
        BatchNormalization(),
        Conv1D(64, kernel_size=(11,), strides=1),
        PReLU(),
        MaxPooling1D(pool_size=pool_size, strides=2),
        Conv1D(64, (1,)),
        PReLU(),
        MaxPooling1D(pool_size=pool_size),
        Flatten(),
        Dense(1024),
        PReLU(),
        Dense(512),
        PReLU(),
        Dense(256),
        PReLU(),
        Dense(128),
        PReLU(),
        Dense(64),
        PReLU(),
        Dense(num_classes, activation='softmax')
    ]
    make_exp_params(exp, layers)
    return Sequential(layers)


def conv_model5(exp=None):
    layers = [
        InputLayer(input_shape=(28, 11)),
        Conv1D(64, 3, activation='relu'),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(3),
        Conv1D(128, 3, activation='relu'),
        Conv1D(128, 3, activation='relu'),
        GlobalAveragePooling1D(),
        # Flatten(),
        Dense(28, activation='softmax')
    ]
    make_exp_params(exp, layers)
    return Sequential(layers)


def dense_model(exp=None):
    layers = [
        #LSTM(28, input_shape=(28, 11,), activation='relu', unroll=True, return_sequences=True),
        #TimeDistributed(Dense(1)),
        InputLayer(input_shape=(28, 11,)),
        BatchNormalization(),
        Dense(512, activation="relu"),
        Dense(256),
        LeakyReLU(),
        Dense(256),
        LeakyReLU(),
        Dense(256),
        LeakyReLU(),
        Dense(1),
        LeakyReLU(),
        Flatten(),
        Dense(PLANET_MAX_NUM, activation='softmax')
    ]
    make_exp_params(exp, layers)
    return Sequential(layers)


def dense_model2(exp=None):
    layers = [
        InputLayer(input_shape=(28, 11,)),
        BatchNormalization(),
        Dense(11, activation='relu'),
        #Dense(11, activation='tanh'),
        Dense(12),
        LeakyReLU(),
        Dropout(0.5),
        Dense(64),
        LeakyReLU(),
        Dense(48),
        LeakyReLU(),
        Dense(1),
        Flatten(),
        Dense(28, activation='softmax')
    ]
    make_exp_params(exp, layers)
    return Sequential(layers)

def dense_model3(exp=None):
    layers = [
        InputLayer(input_shape=(28, 11,)),
        BatchNormalization(),
        Dense(12, activation='tanh'),
        Dense(64),
        LeakyReLU(),
        Dropout(0.5),
        Dense(64),
        LeakyReLU(),
        Dense(48),
        LeakyReLU(),
        Dropout(0.2),
        Dense(1, activation=None),
        Flatten(),
        Dense(28, activation='softmax')
    ]
    model = Sequential(layers)
    make_exp_params(exp, model)
    return model

def new_model(exp=None):
    num_classes = PLANET_MAX_NUM
    num_features = PER_PLANET_FEATURES
    layers = [
        InputLayer(input_shape=(28, 11)),  # batch_input_shape=(None, 28, 11)),
        Reshape((-1, 11)),
        Dense(128),
        PReLU(),
        Dense(64),
        PReLU(),
        Dense(11, activation='tanh'),
        #PReLU(),
        Reshape((-1, 28)),
        Flatten(),
        Masking(),
        Dense(num_classes, activation='softmax')
    ]
    make_exp_params(exp, layers)
    return Sequential(layers)


def lstm(exp=None):
    layers = [
        Input(shape=(28, 11)),
        LSTM(11, activation='relu', return_sequences=True, go_backwards=True),
        LSTM(28, activation='sigmoid', return_sequences=True, go_backwards=True),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(28, activation='softmax'),
    ]
    make_exp_params(exp, layers)
    inputs = layers[0]
    lstm1 = layers[1](inputs)
    lstm2 = layers[2](lstm1)
    flat = layers[3](lstm2)
    dense = layers[4](flat)
    predictions = layers[5](dense)
    model = Model(inputs=inputs, outputs=predictions)
    return model


def lstm2(exp=None):
    layers = [
        Input(shape=(28,11)),
        LSTM(64, activation='relu', return_sequences=True, go_backwards=True),
        LSTM(32, return_sequences=True, go_backwards=True),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(28, activation='softmax')
    ]
    make_exp_params(exp, layers)
    inputs = layers[0]
    lstm1 = layers[1](inputs)
    lstm2 = layers[2](lstm1) #(middle)
    flat = layers[3](lstm2)
    dense = layers[4](flat)
    preds = layers[5](dense)
    model = Model(inputs=inputs, outputs=preds)
    return model


def func_model(exp=None):
    layers = [
        Input(shape=(28, 11)),
        Dense(48, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation=None),
        Flatten(),
        Dense(11, activation='linear'),
        Dense(28, activation='softmax')
    ]
    make_exp_params(exp, layers)
    inputs = layers[0]
    layer1 = layers[1](inputs)
    layer2 = layers[2](layer1)
    layer3 = layers[3](layer2)
    flattened = layers[4](layer3)
    logits = layers[5](flattened)
    prediction = layers[6](logits)
    model = Model(inputs=inputs, outputs=prediction)
    return model


def stacked_model(exp=None):
    layers = [
        InputLayer(input_shape=(28,11)),
        LSTM(32, return_sequences=True),
        LSTM(32, return_sequences=True),
        LSTM(32),
        Dense(28, activation="softmax")
    ]
    make_exp_params(exp, layers)
    return Sequential(layers)


class Hyperdash(Callback):
    def __init__(self, exp, model):
        super(Hyperdash, self).__init__()
        self.exp = exp
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        cur_lr = K.get_value(self.model.optimizer.lr)
        self.exp.metric("lr", cur_lr)
        val_acc = logs.get('val_acc')
        val_loss = logs.get('val_loss')
        acc = logs.get("acc")
        loss = logs.get("loss")
        if epoch is not None:
            self.exp.metric("epoch", epoch + 1)
        if acc is not None:
            self.exp.metric("acc", acc)
        if loss is not None:
            self.exp.metric("loss", loss)
        if val_acc is not None:
            self.exp.metric("val_acc", val_acc)
        if val_loss is not None:
            self.exp.metric("val_loss", val_loss)

def live_model(exp=None):
    layers = [
        InputLayer(input_shape=(28, 11,)),
        Dense(11, activation='relu'),
        BatchNormalization(),
        Dense(12),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64),
        LeakyReLU(),
        BatchNormalization(),
        Dense(48),
        LeakyReLU(),
        BatchNormalization(),
        Dense(1),
        BatchNormalization(),
        Flatten(),
        Dense(28, activation='softmax')
    ]
    model = Sequential(layers)
    make_exp_params(exp, model)
    return model

MODELS = {
    "dense": dense_model,
    "dense2": dense_model2,
    "dense3": dense_model3,
    "conv": conv_model,
    "conv2": conv_model2,
    "conv4": conv_model4,
    "conv3": conv_model3,
    "conv5": conv_model5,
    "new": new_model,
    "func": func_model,
    "lstm": lstm,
    "lstm2": lstm2,
    "evo": evo_model,
    "stacked": stacked_model,
    "deathbot2": live_model,
    "deathbot": dense_model3
}


class KerasModel:
    def __init__(self, name='deathbot', load_weights=False, training=False, batch_size=100, lr=1e-3, location=None):
        self.session = tf.Session()
        self.name = name

        if training: 
            from hyperdash import Experiment
            self.exp = Experiment(name)

        if name in MODELS.keys(): self.model = MODELS[name]() if not training else MODELS[name](self.exp)
        adam = Adam(lr=lr)
        nadam = Nadam(lr=lr)
        #rms = RMSprop(lr=lr)
        #sgd = SGD(lr=lr)
        self.optimizer = adam if name == "evo" else nadam
        loss = ["binary_crossentropy", "categorical_crossentropy", "poisson"]
        self.model.compile(optimizer=self.optimizer, loss=loss[1],
                           metrics=["acc"])

        self.callbacks = []
        if training:
            self.exp.param("lr", lr)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=4, min_lr=1e-4, verbose=1)
            tb = TensorBoard('./models/logs/{}'.format(name), write_graph=True)
            cp = ModelCheckpoint(filepath='./models/weights-{}.hdf5'.format(name), monitor='val_acc', verbose=1,
                                 save_best_only=True)
            hd = Hyperdash(self.exp, self.model)
            es = EarlyStopping('val_acc', patience=5, verbose=1)
            self.callbacks = [cp, tb, hd, reduce_lr, es]

        if load_weights:
            #print(os.listdir(os.getcwd()))
            self.model.load_weights('./deathbot/weights-{}.hdf5'.format(name))
            if training: print('Weights Loaded...')

    def save(self, path):
        self.model.save(path + self.name + ".h5")

    def fit(self, input_data, expected_output_data, batch_size=100, epochs=1):
        input_data = self.normalize_input(input_data)
        return self.model.fit(input_data,
                              expected_output_data,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              callbacks=self.callbacks,
                              validation_split=0.2,
                              shuffle=False)

    def predict(self, input_data, batch_size=1, p=False):
        return list(map(self.clean_pred, self.model.predict(self.normalize_input(np.array([input_data])), batch_size=batch_size)[0]))

    def compute_loss(self, input_data, expected_output_data):
        return self.model.evaluate(self.normalize_input(input_data), expected_output_data, batch_size=1, verbose=1)

    @staticmethod
    def clean_pred(pred):
        return pred if pred > 0.01 else 0.0

    @staticmethod
    def normalize_input(input_data):
        # Assert the shape is what we expect
        assert len(input_data.shape) == 3 and input_data.shape[1] == PLANET_MAX_NUM and input_data.shape[2] == PER_PLANET_FEATURES
        m = np.expand_dims(input_data.mean(axis=1), axis=1)
        s = np.expand_dims(input_data.std(axis=1), axis=1)
        return (input_data - m) / (s + 1e-6)
