from anathema import net as anet

import hlt
import logging
import math, random
import numpy
import torch
import platform, sys, os
import manager_constants

def run_game(network, bot_strings=None):
    """
    Runs a game and returns the input/output history of the winner of the game in tensor form.
    :param bot_strings: ["java botname", "python botname", ...]
    :return: the input/output history of the winner of the game in tensor form e.g. states, outputs = run_game(...)
    """
    pass

model_string = None



if len(sys.argv) < 3:

    exit("Need a model file, and at least one run command.")


model_string = sys.argv[1]
net = anet.Net()
if os.path.isfile(model_string):
    net = torch.load(model_string)

games_played = int(model_string.split("-")[1])

out_string = model_string.split("-")[0]


while True:

    state_history = []
    move_history = []

    for game_id in range(0, manager_constants.rollout_games):

        states, outputs = run_game(net)

        state_history.append(states)
        move_history.append(outputs)

    games_played += manager_constants.rollout_games

    net.my_train(torch.Tensor(states), torch.Tensor(outputs), epochs=10)


    torch.save(net, "{} {}".format(out_string, games_played))
