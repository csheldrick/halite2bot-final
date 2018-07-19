import argparse
import json
import os.path
import zipfile
from random import shuffle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from parsing import parse, load_data
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from model import KerasModel, dense_model, conv_model, conv_model3
from common import PLANET_MAX_NUM, PER_PLANET_FEATURES
from devol import DEvol, BotGenomeHandler


def fetch_data_gen(dir, limit):
    replay_files = sorted(
        [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.startswith('replay-')])
    print("Found {} games".format(len(replay_files)))
    loaded_games = 0
    all_data = []
    for i in range(len(replay_files)):
        full_path = os.path.join(dir, replay_files[i])
        with open(full_path, encoding='utf-8') as game:
            game_data = game.read()
        if game_data[0:2] == "b'":
            game_data = game_data[2:-1]
        game_data_json = json.loads(game_data)
        all_data.append(game_data_json)
        loaded_games += 1
        if loaded_games == limit:
            yield all_data
            loaded_games = 0
            all_data = []


def fetch_data_dir(dir, limit):
    replay_files = sorted(
        [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.startswith('replay-')])
    shuffle(replay_files)
    print("Found {} games".format(len(replay_files)))
    if len(replay_files) == 0: raise Exception("Didn't find any game replays.")
    print("Trying to load up to {} games".format(limit))
    loaded_games = 0
    all_data = []
    for r in replay_files:
        full_path = os.path.join(dir, r)
        with open(full_path, encoding="utf-8") as game:
            game_data = game.read()
            if game_data[0:2] == "b'":
                game_data = game_data[2:-1]
            game_json_data = json.loads(game_data)
            all_data.append(game_json_data)
        loaded_games += 1

        if loaded_games >= limit:
            break
    print("{} games loaded".format(loaded_games))
    return all_data


def fetch_data_zip(zipfilename, limit):
    all_jsons = []
    with zipfile.ZipFile(zipfilename) as z:
        print("Found {} games.".format(len(z.filelist)))
        print("Trying to load up to {} games ...".format(limit))
        for i in z.filelist[:limit]:
            with z.open(i, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                if lines[0][0:2] == "b’":
                    lines[0] = lines[0][2:-1]
                d = json.loads(lines[0].decode())
                all_jsons.append(d)
    print("{} games loaded.".format(len(all_jsons)))
    return all_jsons


# Normalize planet features within each frame.
def normalize_input(input_data):
    # Assert the shape is what we expect
    shape = input_data.shape
    assert len(shape) == 3 and shape[1] == PLANET_MAX_NUM and shape[2] == PER_PLANET_FEATURES

    m = np.expand_dims(input_data.mean(axis=1), axis=1)
    s = np.expand_dims(input_data.std(axis=1), axis=1)
    return (input_data - m) / (s + 1e-6)


def generator(features, labels, batch_size):
    i = 0
    while True:
        s = i * batch_size % len(features)
        e = s + batch_size
        batch_features = features[s:e]
        batch_labels = labels[s:e]
        yield normalize_input(batch_features), batch_labels
        i += 1
        if i >= batch_size: i = 0


def summary(history, location):
    # list all data in history
    print(', '.join(history.history.keys()))
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel(['accuracy', 'val accuracy'])
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(location + 'accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel(['loss', 'val loss'])
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(location + 'loss.png')


def get_models():
    models = [dense_model(), conv_model(), conv_model2(), conv_model3()]
    for model in models:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return models


def get_model():
    model = conv_model2()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def evaluate(input_data, target_output):
    estimator = KerasClassifier(build_fn=get_model, epochs=200, batch_size=5, verbose=1)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, input_data, target_output, cv=kfold)
    name = 'Conv2D'
    print("{0} Baseline: {1:.2f} {2:.2f}".format(name, results.mean() * 100, results.std() * 100))


def main():
    sess = tf.Session()
    with sess.as_default():
        parser = argparse.ArgumentParser(description="Halite 2 ML Training")
        parser.add_argument("--model_name", help="Name of the model", default="keras-model.h5")
        parser.add_argument("--minibatch_size", help="Size of the minibatch", default=100, type=int)
        parser.add_argument("--steps", help="Number of steps", default=1000, type=int)
        parser.add_argument("--games_limit", help="Number of games", default=1000, type=int)
        parser.add_argument("--data", help="Location of Replays", default="data/sample/")
        parser.add_argument("--cache", help="Model to Load", default=None)
        parser.add_argument("--load_data", help="Load Features from file", default=True, type=bool)
        parser.add_argument("--pack", help="Which replay pack to use", default="all")
        parser.add_argument("--load_weights", help="Load weights", default=False, type=bool)
        parser.add_argument("--lr", help="Learning Rate", default=1e-3, type=float)
        parser.add_argument("--evo", help="use Genetic evolution", default=False, type=bool)
        args = parser.parse_args()
        if not args.load_data:
            if args.seed:
                np.random.seed(args.seed)
            if args.data.endswith('.zip'):
                raw_data = fetch_data_zip(args.data, args.games_limit)
            else:
                raw_data = fetch_data_dir(args.data, args.games_limit)
            data_input, data_output = parse(raw_data, None, args.dump_features_location)
        else:
            data_input, data_output = load_data(pack=args.pack)
        data_size = len(data_input)
        training_input, training_output = data_input, data_output

        training_data_size = len(training_input)
        # randomly permute the data
        permutation = np.random.permutation(training_data_size)
        training_input, training_output = training_input[permutation], training_output[permutation]

        if not args.evo:
            kmodel = KerasModel(args.model_name, args.load_weights, training=True,
                                batch_size=args.minibatch_size, lr=args.lr)
            model = kmodel.model
            model.summary()
            eval_input = kmodel.normalize_input(training_input)
            for i in range(10):
                preds = kmodel.predict(training_input[i])
                print("Pred {}".format(preds))
                count = 0
                true_count = 0
                for i, v in enumerate(preds):
                    count += 1
                    as_perc = round(v, 3)*100
                    t_as_perc = round(training_output[0][i], 3)*100
                    if as_perc == t_as_perc: true_count += 1
                    print("{0:.2f} vs {1:.2f} | {2}".format(as_perc, t_as_perc, as_perc == t_as_perc))
                print("{0}/{1} = {2:.2f}%".format(true_count, count, true_count/count*100))

            score = model.evaluate(eval_input, training_output, verbose=1)
            print("\nInitial: loss: {0:.2f}, acc: {1:.2f}%".format(score[0], score[1] * 100))
            print("Metrics: {}".format(model.metrics_names))
            history = kmodel.fit(training_input, training_output, batch_size=args.minibatch_size, epochs=args.steps)

            current_directory = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_directory, os.path.pardir, "models/")
            kmodel.save(model_path)
            summary(history, model_path)
        else:
            max_conv_layers = 2
            max_dense_layers = 4
            max_conv_kernels = 128
            max_dense_nodes = 512
            input_shape = data_input.shape[1:]
            num_classes = 28
            genome_handler = BotGenomeHandler(max_conv_layers, max_dense_layers, max_conv_kernels, max_dense_nodes, input_shape, num_classes)
            num_generations = 20
            population_size = 30
            num_epochs = 1
            devol = DEvol(genome_handler)
            perc = int(training_data_size * .8)

            x_train, x_test = training_input[perc:], training_input[:perc]
            y_train, y_test = training_output[perc:], training_output[:perc]

            dataset = ((x_train, y_train), (x_test, y_test))
            model, accuracy, loss = devol.run(dataset, num_generations, population_size, num_epochs)
            model.summary()
            print("Accuracy: {}\tLoss: {}".format(accuracy, loss))



if __name__ == "__main__":
    main()

