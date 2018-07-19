#!/usr/bin/python
import argparse
import json
import os.path
import zipfile
from random import choice, shuffle, randint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd

from final.parsing import parse, load_data

from final.neural_net import NeuralNet, NeuralNetOrig



def fetch_data_dir(directory, limit):
    """
    Loads up to limit games into Python dictionaries from uncompressed replay files.
    """
    replay_files = list([f for f in os.listdir(directory) if
                           os.path.isfile(os.path.join(directory, f)) and f.startswith("replay-")])
    [shuffle(replay_files) for _ in range(randint(0,10))]
    print("Found {} games.".format(len(replay_files)))
    if len(replay_files) == 0:
        raise Exception("Didn't find any game replays. Please call make games.")

    print("Trying to load up to {} games ...".format(limit))

    loaded_games = 0

    all_data = []
    for r in replay_files:
        full_path = os.path.join(directory, r)
        with open(full_path, encoding="utf-8") as game:
            game_data = game.read()
            if game_data[0:2] == "b'":
                game_data = game_data[2:-1]
            game_json_data = json.loads(game_data)
            all_data.append(game_json_data)
        loaded_games = loaded_games + 1

        if loaded_games >= limit:
            break

    print("{} games loaded.".format(loaded_games))

    return all_data


def fetch_data_zip(zipfilename, limit):
    """
    Loads up to limit games into Python dictionaries from a zipfile containing uncompressed replay files.
    """
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


def main():
    parser = argparse.ArgumentParser(description="Halite II training")
    parser.add_argument("--model_name", help="Name of the model", default="test-model")
    parser.add_argument("--minibatch_size", type=int, help="Size of the minibatch", default=100)
    parser.add_argument("--steps", type=int, help="Number of steps in the training", default=500)
    parser.add_argument("--data", help="Data directory or zip file containing uncompressed games")
    parser.add_argument("--cache", help="Location of the model we should continue to train")
    parser.add_argument("--games_limit", type=int, help="Train on up to games_limit games", default=1000)
    parser.add_argument("--seed", type=int, help="Random seed to make the training deterministic")
    parser.add_argument("--bot_to_imitate", help="Name of the bot whose strategy we want to learn")
    parser.add_argument("--pack", help="Which replay pack to use")
    parser.add_argument("--network", help="Network Model", default="NeuralNet")
    parser.add_argument("--load_data", help="Load Features from file", default=True, type=bool)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=1e-4)
    args = parser.parse_args()

    # Make deterministic if needed
    if args.seed is not None:
        np.random.seed(args.seed)

    nn = NeuralNet(name=args.model_name, cached_model=args.cache, seed=args.seed, lr=args.lr, training=True)
    if not args.load_data:
        if args.data.endswith('.zip'):
            raw_data = fetch_data_zip(args.data, args.games_limit)
        else:
            raw_data = fetch_data_dir(args.data, args.games_limit)

        data_input, data_output = parse(raw_data, args.bot_to_imitate, args.dump_features_location)
    else:
        data_input, data_output = load_data(pack=args.pack)
    data_size = len(data_input)
    print(data_size == len(data_output))
    training_input, training_output = data_input[:int(0.85 * data_size)], data_output[:int(0.85 * data_size)]
    validation_input, validation_output = data_input[int(0.85 * data_size):], data_output[int(0.85 * data_size):]

    training_data_size = len(training_input)
    print(training_data_size == len(training_output))
    print(len(validation_input) == len(validation_output))
    # randomly permute the data
    permutation = np.random.permutation(training_data_size)
    training_input, training_output = training_input[permutation], training_output[permutation]
    l = nn.compute_loss(validation_input, validation_output)
    print("Initial, cross validation loss: {}".format(l))
    preds = nn.predict(validation_input[0])
    for p in preds: print(p)
    curves = []
    for s in range(args.steps):
        start = (s * args.minibatch_size) % training_data_size
        end = start + args.minibatch_size
        training_loss = nn.fit(training_input[start:end], training_output[start:end])
        if s % 50 == 0 or s == args.steps - 1:
            validation_loss = nn.compute_loss(validation_input, validation_output)
            print("Step: {0}, cv loss: {1:.2f}, training_loss: {2:.2f}".format(s, validation_loss, training_loss))
            curves.append((s, training_loss, validation_loss))

    cf = pd.DataFrame(curves, columns=['step', 'training_loss', 'cv_loss'])
    fig = cf.plot(x='step', y=['training_loss', 'cv_loss']).get_figure()

    # Save the trained model, so it can be used by the bot
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, os.path.pardir, "models", args.model_name + ".ckpt")
    print("Training finished, serializing model to {}".format(model_path))
    nn.save(model_path)
    print("Model serialized")

    curve_path = os.path.join(current_directory, os.path.pardir, "models", args.model_name + "_training_plot.png")
    fig.savefig(curve_path)



if __name__ == "__main__":
    main()
