import os
import tensorflow as tf
import numpy as np
from final.common import PLANET_MAX_NUM, PER_PLANET_FEATURES

# We don't want tensorflow to produce any warnings in the standard output, since the bot communicates
# with the game engine through stdout/stdin.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


# Normalize planet features within each frame.
def normalize_input(input_data):
    # Assert the shape is what we expect
    shape = input_data.shape
    assert len(shape) == 3 and shape[1] == PLANET_MAX_NUM and shape[2] == PER_PLANET_FEATURES

    m = np.expand_dims(input_data.mean(axis=1), axis=1)
    s = np.expand_dims(input_data.std(axis=1), axis=1)
    return (input_data - m) / (s + 1e-6)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


flatten = tf.contrib.layers.flatten
fully_connected = tf.contrib.layers.fully_connected
layer_norm = tf.contrib.layers.layer_norm
dropout = tf.nn.dropout
tanh = tf.nn.tanh
relu = tf.nn.relu
relu6 = tf.nn.relu6
variance_scaling_initializer = tf.contrib.layers.variance_scaling_initializer


def multilabel_accuracy(inputs, targets):
    acc = 0.0
    for i, v in enumerate(inputs):
        rv = round(v, 3)
        if rv < 0.01:
            rv = 0.0
        tv = round(targets[i], 3)
        if rv == tv:
            acc += 1 / 28
    return acc * 100


class NeuralNet(object):
    LAYER1_SIZE = 522  # 12
    LAYER2_SIZE = 256  # 6
    LAYER3_SIZE = 128
    LAYER4_SIZE = 64
    LAYER5_SIZE = 32
    OUTPUT_SIZE = 1

    def __init__(self, name='nn-model', cached_model=None, seed=None, lr=1e-4, training=False):
        self.graph = tf.Graph()
        self.training = training
        if self.training:
            from hyperdash import Experiment

            self.exp = Experiment(name)

        with self.graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self.session = tf.Session()
            self.features = tf.placeholder(dtype=tf.float32, name="input_features",
                                           shape=(None, PLANET_MAX_NUM, PER_PLANET_FEATURES))
            # target_distribution describes what the bot did in a real game.
            # For instance, if it sent 20% of the ships to the first planet and 15% of the ships to the second planet,
            # then expected_distribution = [0.2, 0.15 ...]
            self.target_distribution = tf.placeholder(dtype=tf.float32, name="target_distribution",
                                                      shape=(None, PLANET_MAX_NUM))
            # Combine all the planets from all the frames together, so it's easier to share
            # the weights and biases between them in the network.
            flattened_frames = tf.reshape(self.features, [-1, PER_PLANET_FEATURES])

            layer1 = fully_connected(flattened_frames, 512)
            layer2 = fully_connected(layer1, 256)
            layer3 = fully_connected(layer2, 128)
            # Group back into frames
            layer4 = fully_connected(layer3, 64)
            layer5 = fully_connected(layer4, 32)
            layer6 = fully_connected(layer5, 1, activation_fn=None)
            logits = tf.reshape(layer6, [-1, PLANET_MAX_NUM])

            self.prediction_normalized = tf.nn.softmax(logits)
            self.loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.target_distribution))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # returns Op

            self.train_op = self.optimizer.minimize(self.loss_op)

            # self.acc_op = tf.reduce_mean(tf.reduce_min(tf.cast(self.prediction_normalized, tf.float32), 1))
            # self.acc, self.update_acc_op = tf.metrics.mean_per_class_accuracy(self.target_distribution, self.prediction_normalized, 28)
            # multilabel_accuracy(self.prediction_normalized, self.target_distribution)
            self.saver = tf.train.Saver()
            if self.training:
                self.exp.param("lr", lr)
            if cached_model is None:
                self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            else:
                self.session.run(tf.local_variables_initializer())
                self.saver.restore(self.session, cached_model)

    def fit(self, input_data, expected_output_data):
        loss, _ = self.session.run([self.loss_op, self.train_op],
                                        feed_dict={self.features: normalize_input(input_data),
                                                   self.target_distribution: expected_output_data})

        if self.training:
            self.exp.metric("training_loss", loss)
        return loss

    def predict(self, input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        return self.session.run(self.prediction_normalized,
                                feed_dict={self.features: normalize_input(np.array([input_data]))})[0]

    def compute_loss(self, input_data, expected_output_data):
        """
        Compute loss on the input data without running any training.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        loss = self.session.run(self.loss_op,
                                     feed_dict={self.features: normalize_input(input_data),
                                                self.target_distribution: expected_output_data})
        if self.training:
            self.exp.metric("val_loss", loss)
        return loss

    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        self.saver.save(self.session, path)


class NeuralNetOrig(object):
    LAYER1_SIZE = 32  # 12
    LAYER2_SIZE = 16  # 6
    OUTPUT_SIZE = 1

    def __init__(self, cached_model=None, seed=None):
        self._graph = tf.Graph()

        with self._graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._session = tf.Session()
            self._features = tf.placeholder(dtype=tf.float32, name="input_features",
                                            shape=(None, PLANET_MAX_NUM, PER_PLANET_FEATURES))
            # target_distribution describes what the bot did in a real game.
            # For instance, if it sent 20% of the ships to the first planet and 15% of the ships to the second planet,
            # then expected_distribution = [0.2, 0.15 ...]
            self._target_distribution = tf.placeholder(dtype=tf.float32, name="target_distribution",
                                                       shape=(None, PLANET_MAX_NUM))
            # Combine all the planets from all the frames together, so it's easier to share
            # the weights and biases between them in the network.
            flattened_frames = tf.reshape(self._features, [-1, PER_PLANET_FEATURES])

            first_layer = tf.contrib.layers.fully_connected(flattened_frames, self.LAYER1_SIZE)
            second_layer = tf.contrib.layers.fully_connected(first_layer, self.LAYER2_SIZE)
            third_layer = tf.contrib.layers.fully_connected(second_layer, 1, activation_fn=tf.nn.relu)
            logits = tf.reshape(third_layer, [-1, PLANET_MAX_NUM])
            self._prediction_normalized = tf.nn.softmax(logits)
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self._target_distribution))

            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self._loss)
            # self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(self._loss)
            self._saver = tf.train.Saver()

            if cached_model is None:
                self._session.run(tf.global_variables_initializer())
            else:
                self._saver.restore(self._session, cached_model)

    def fit(self, input_data, expected_output_data):
        """
        Perform one step of training on the training data.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        loss, _ = self._session.run([self._loss, self._optimizer],
                                    feed_dict={self._features: normalize_input(input_data),
                                               self._target_distribution: expected_output_data})

        return loss

    def predict(self, input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        return self._session.run(self._prediction_normalized,
                                 feed_dict={self._features: normalize_input(np.array([input_data]))})[0]

    def compute_loss(self, input_data, expected_output_data):
        """
        Compute loss on the input data without running any training.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        return self._session.run(self._loss,
                                 feed_dict={self._features: normalize_input(input_data),
                                            self._target_distribution: expected_output_data})

    def accuracy(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy

    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        self._saver.save(self._session, path)


class NeuralNetAdam(object):
    LAYER1_SIZE = 32  # 12
    LAYER2_SIZE = 16  # 6
    OUTPUT_SIZE = 1

    def __init__(self, cached_model=None, seed=None):
        self._graph = tf.Graph()

        with self._graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._session = tf.Session()
            self._features = tf.placeholder(dtype=tf.float32, name="input_features",
                                            shape=(None, PLANET_MAX_NUM, PER_PLANET_FEATURES))
            # target_distribution describes what the bot did in a real game.
            # For instance, if it sent 20% of the ships to the first planet and 15% of the ships to the second planet,
            # then expected_distribution = [0.2, 0.15 ...]
            self._target_distribution = tf.placeholder(dtype=tf.float32, name="target_distribution",
                                                       shape=(None, PLANET_MAX_NUM))
            # Combine all the planets from all the frames together, so it's easier to share
            # the weights and biases between them in the network.
            flattened_frames = tf.reshape(self._features, [-1, PER_PLANET_FEATURES])

            first_layer = tf.contrib.layers.fully_connected(flattened_frames, self.LAYER1_SIZE)
            second_layer = tf.contrib.layers.fully_connected(first_layer, self.LAYER2_SIZE)
            third_layer = tf.contrib.layers.fully_connected(second_layer, 1, activation_fn=tf.nn.relu)
            logits = tf.reshape(third_layer, [-1, PLANET_MAX_NUM])
            self._prediction_normalized = tf.nn.softmax(logits)
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self._target_distribution))

            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self._loss)

            self._saver = tf.train.Saver()

            if cached_model is None:
                self._session.run(tf.global_variables_initializer())
            else:
                self._saver.restore(self._session, cached_model)

    def fit(self, input_data, expected_output_data):
        """
        Perform one step of training on the training data.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        loss, _ = self._session.run([self._loss, self._optimizer],
                                    feed_dict={self._features: normalize_input(input_data),
                                               self._target_distribution: expected_output_data})

        return loss

    def predict(self, input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        return self._session.run(self._prediction_normalized,
                                 feed_dict={self._features: normalize_input(np.array([input_data]))})[0]

    def compute_loss(self, input_data, expected_output_data):
        """
        Compute loss on the input data without running any training.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        return self._session.run(self._loss,
                                 feed_dict={self._features: normalize_input(input_data),
                                            self._target_distribution: expected_output_data})

    def accuracy(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy

    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        self._saver.save(self._session, path)
