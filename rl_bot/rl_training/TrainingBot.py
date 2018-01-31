from pprint import pprint
import hlt
import manager_constants
import numpy as np
from rl_bot import newnet as anet
from rl_bot import lstmnet as bnet
import logging
from math import sqrt, exp, degrees
import random, time
import subprocess
import sys
import keras.backend as K
import platform, os

NUM_PLAYERS = 2
NUM_GAMES = 1000

NUM_FEATURES = 8
NUM_OUTPUT_FEATURES = 3

def clamp(value, small, large):
    return max(min(value, large), small)

def sigmoid(v):
    return 1 / (1+exp(-v))

def distance2(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def distance(x1, y1, x2, y2):
    return sqrt(distance2(x1, y1, x2, y2))

def normalize_input(input_data):
    m = np.expand_dims(input_data.mean(axis=-1), axis=-1)
    s = np.expand_dims(input_data.std(axis=-1), axis=-1)
    return (input_data - m) / (s + 1e-6)

def convert_map_to_tensor(game_map, input_tensor, my_ships, cmd_history):
    my_ships.clear()
    me = game_map.get_me()
    # feature vector: 
    # ship: [ship hp, ship friendliness, docking status
    # planet: planet hp, planet size, % docked_ships, planet friendliness]
    for player in game_map.all_players():
        owner_feature = 1 if player.id == game_map.my_id else -1
        for ship in player.all_ships():
            x = min(round(ship.x), game_map.width - 1)
            y = min(round(ship.y), game_map.height - 1)
            # hp from [0, 1]
            input_tensor[0][x][y][0] = (ship.health/255)*100 #/ (ship. * 255.0) 
            # friendless: 0 if me, 1 if enemy
            input_tensor[0][x][y][1] = owner_feature
            # 0 if docked, .33 if docking, .66 if undocking, 1 if  undocked
            dock = ship.docking_status.value
            if dock == 0: dock = 1.0
            elif dock == 1: dock = .33
            elif dock == 2: dock = 0.0
            elif dock == 3: dock = .66
            input_tensor[0][x][y][2] = dock
            #closest_planet = ship.closest_planet(game_map)
            #angle = ship.calculate_angle_between(closest_planet)/360.0
            #dist = ship.calculate_distance_between(closest_planet)
            if len(cmd_history):
                past_cmds = cmd_history[-1]
                last = [c for c in past_cmds if c.split(" ")[0] == "t" and int(c.split(" ")[1]) == ship.id]
                if len(last):
                    last_angle = int(last[0].split(" ")[-1]) #angle/360.0
                    input_tensor[0][x][y][3] = last_angle #if last_angle > 0 else random.randint(0,360)
            #input_tensor[0][x][y][3] = clamp(1.0 - float(dist/game_map.width), 0.01, 1)
            #print(dist, input_tensor[0][x][y][3])
            if owner_feature == 1:
                my_ships[(x, y)] = ship

    for planet in game_map.all_planets():
        x = int(planet.x)
        y = int(planet.y)
        # hp from [0, 1]
        hp = planet.health / (planet.radius * 255)
        input_tensor[0][x][y][4] = hp
        # owner of this planet: 1 if me, .5 if enemy, 0.01 if unowned
        ownership = (1 if planet.owner == game_map.my_id else -1) if planet.is_owned() else -1
        input_tensor[0][x][y][5] = ownership
        # radius from [0, 1]
        input_tensor[0][x][y][6] = (planet.radius - 3) / 5
        # % of docked ships [0, 1]
        input_tensor[0][x][y][7] = (len(planet.all_docked_ships()) / planet.num_docking_spots)
    #pprint(input_tensor[0][input_tensor[0]>0])
    #input_tensor = normalize_input(input_tensor)
    #for ship in my_ships:
        #print(input_tensor[0][ship[0]][ship[1]])

def scale(x):
    return (x / 99.0) * 2.0 - 1.0

def one_or_negative_one(v=None):
    if random.random() > .5 and v:
        return 1 if v > .5 else -1
    else: return 1 if random.random() > .5 else -1

def skew_towards_zero(v=None):
    if random.random() > .5 and v:
        return (1 - sqrt(sqrt(1 - v)))
    else: return (1 -sqrt(sqrt(1 - random.random())))


def run_game(num_players, net):
    """
    Runs a single game against itself. Uses the same network to calculate moves for EACH player in this game.
    :param num_players: Number of players to simulate during this game (2-4)
    :return: n/a
    """
    # initialize halite
    run_commands = []
    for i in range(num_players):
        run_commands.append("./fake_bot2 {}".format(i))
    #run_commands.append("python ../SettlerBot/MyBot.py")
    #run_commands.append("python ../SettlerBot/MyBot.py")
    w = 40*3
    h = 40*2
    subprocess.Popen(["./halite", "-t", "-d", "{} {}".format(w,h)] + run_commands)
    # GAME START
    test = np.zeros((1, w, h, NUM_OUTPUT_FEATURES))

    games_per_player = []
    maps_per_player = []
    board_states_per_player = []
    outputs_per_player = []
    ships_per_player = []
    old_ships = []
    eliminated = []

    from_halite_fifos = []
    to_halite_fifos = []

    made_ships = [False for _ in range(num_players)]

    for i in range(num_players):
        from_halite_fifos.append(os.fdopen(os.open("pipes/from_halite_{}".format(i), os.O_RDONLY), "r"))
        to_halite_fifos.append(open("pipes/to_halite_{}".format(i), "w"))
        games_per_player.append(hlt.Game("Anathema{}".format(i), from_halite_fifos[i], to_halite_fifos[i]))
        print("Starting << anathema >> for player {}".format(i))
        outputs_per_player.append([])
        board_states_per_player.append([])
        ships_per_player.append({})
        maps_per_player.append(None)
        eliminated.append(False)
        old_ships.append([])
    # Initialize zeroed input/output tensors
    input_tensor = np.zeros((1, games_per_player[i].map.width, games_per_player[i].map.height, NUM_FEATURES))
    output_tensor = np.zeros((1, games_per_player[i].map.width, games_per_player[i].map.height, NUM_OUTPUT_FEATURES))
    turns = 0
    output_history = [[] for _ in range(num_players)]
    input_history = [[] for _ in range(num_players)]
    cmd_history = [[] for _ in range(num_players)]
    decay = 0.001
    eps = 1.0
    while True:
        turns += 1
        eps -= decay
        # play out each player's turn
        for i, game in enumerate(games_per_player):
            if eliminated[i] is True:
                continue
            # need a way to detect when this player has lost and shouldnt be updated anymore
            try:
                game_map = game.update_map()
            except ValueError as e:
                # this player is done playing
                print("Anathema{} eliminated".format(i))
                eliminated[i] = True

                from_halite_fifos[i].close()
                to_halite_fifos[i].close()
                breaker = all(eliminated)
                if breaker:
                    print("all eliminated")
                    if made_ships[i]:
                        return board_states_per_player[i], outputs_per_player[i]
                    else:
                        x = 0 if i == 1 else 1
                        if made_ships[x]:
                            return board_states_per_player[x], outputs_per_player[x]
                        else:
                            return [], []
                continue
            command_queue = []
            my_ships = ships_per_player[i]
            if len(my_ships.keys()) > 3:
                made_ships[i] = True
            #else: made_ships[i] = False
            input_tensor = np.zeros((1, games_per_player[i].map.width, games_per_player[i].map.height, NUM_FEATURES))
            output_tensor = np.zeros((1, games_per_player[i].map.width, games_per_player[i].map.height, NUM_OUTPUT_FEATURES))
            # Rebuild our input tensor based on the map state for this turn
            convert_map_to_tensor(game_map, input_tensor, my_ships, cmd_history[i])
            move_commands = net.model.predict(input_tensor)
            #if i == 0 and np.array_equal(move_commands, test): print("predicted all 0")
            move_commands = move_commands[0]
            fit = False
            for (x, y), this_ship in my_ships.items():
                eps -= decay
                angle, speed, dock = move_commands[x][y].data
                output_tensor[0][x][y][0] = angle
                #if i == 0: print(angle, np.cos(angle), np.arccos(angle))
                command_angle = degrees(np.arccos(angle))
                #command_angle = angle # int(360 * angle) % 360

                if speed == 0: speed += 1.0 - skew_towards_zero()
                # .01 = 0 .1 = 1 .2 = 2 .3 = 3 .4 = 4 .5 = 6 .6+ = 7
                command_speed = np.clip(int(12 * speed), 0, 7)

                output_tensor[0][x][y][2] = dock
                command_dock = dock
                if i == 0:
                    pid = getattr(this_ship.planet, "id") if hasattr(this_ship, "planet") and this_ship.planet else None
                    print("ship: {0} a:{1:.3f} ca:{2:.3f}  s:{3:.3f} cs:{4:.3f} status: {5}  p: {5}".format(this_ship.id, angle,command_angle,speed,command_speed,str(this_ship.docking_status).split(".")[-1], pid))
                closest_planet = this_ship.closest_planet(game_map)
                if not this_ship.can_dock(closest_planet): # and closest_planet.is_full():
                    p = game_map.all_planets()
                    p = [pl for pl in p if not pl.is_full()]
                    p = sorted(p, key=lambda pl: this_ship.calculate_distance_between(pl))
                    closest_planet = p[0]
                dist = this_ship.calculate_distance_between(closest_planet)
                if dist < 7:
                    speed = dist
                    output_tensor[0][x][y][1] = speed
                    command_speed = speed
                tangle = this_ship.calculate_angle_between(closest_planet)
                trad = np.cos(np.deg2rad(tangle))
                # Execute ship command
                if command_dock < 0.1:
                    # we want to undock
                    if this_ship.docking_status.value == this_ship.DockingStatus.DOCKED.value:
                        output_tensor[0][x][y][2] = dock + (one_or_negative_one() + skew_towards_zero())
                        #command_queue.append(this_ship.undock())
                    else:
                        # ship wants to undock, but isnt docked - set output for dock to dock + 1.0 to ensure its > 0.1
                        output_tensor[0][x][y][2] = dock + (1.0 - skew_towards_zero())
                        command_queue.append(this_ship.thrust(command_speed, command_angle))
                else:
                    # we want to dock
                    if this_ship.docking_status.value == this_ship.DockingStatus.UNDOCKED.value:
                        if this_ship.can_dock(closest_planet):
                            # Dock at closest Planet
                            command_queue.append(this_ship.dock(closest_planet))
                        else:
                            # Not in range
                            #tangle = this_ship.calculate_angle_between(closest_planet)/360.0
                            # Squash tangle to [0-1]
                            output_tensor[0][x][y][0] = trad
                            dist = this_ship.calculate_distance_between(closest_planet)
                            if dist > 7 and command_speed < 7:
                                # We want to dock, but the closest planet is more than 7 units away, so we should always output full speed
                                output_tensor[0][x][y][1] = speed + (1.0 - skew_towards_zero())
                            command_queue.append(this_ship.thrust(command_speed, command_angle))
                    else:
                        # Want to dock, but are already docked
                        # Since we are already docked, set angle to 0.0, speed to 0.0 (for loss)
                        #tangle = this_ship.calculate_angle_between(closest_planet)/360.0

                        output_tensor[0][x][y][0] = tangle
                        #output_tensor[0][x][y][1] = 0.0
            board_states_per_player[i].append(input_tensor[0])
            #if np.array_equal(output_tensor[0],test[0]): print("output all 0")
            outputs_per_player[i].append(output_tensor[0])
            cmd_history[i].append(command_queue)
            # Send our set of commands to the Halite engine for this turn
            game.send_command_queue(command_queue)


def get_model_file_arg():

    if len(sys.argv) > 2:
        print("Too many arguments!")
        print("python TrainingBot.py [model file]")
        exit(-1)

    return sys.argv[1] if len(sys.argv) == 2 else None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Halite II training")                       
    parser.add_argument("--model_name", help="Name of the model", default="conv")
    parser.add_argument("--load", action="store_true")
    # load model from file
    #model_file = get_model_file_arg()
    args = parser.parse_args()
    if "lstm" in args.model_name:
        net = bnet.KerasNet()
    else:
        net = anet.KerasNet() #torch.load(model_file)
    if args.load:
        net.model.load_weights("models/{}".format(args.model_name))
    file_prefix, games_played = args.model_name.split("-")
    games_played = int(games_played)

    count = 0
    while True:
        count += 1
        states, outputs = [], []

        print("Game ID:", games_played)
        for game_id in range(0, manager_constants.rollout_games):
            tstates, toutputs = run_game(NUM_PLAYERS, net)
            states += tstates
            outputs += toutputs
            print('Training data', len(states), len(outputs))
            games_played += 1
            #if count == 1 and len(states):
                #net.my_train(states, outputs, epochs=10)


        if len(states) > 0:
            net.my_train(states, outputs, epochs=5)
            print("Saving")
            net.model.save("models/{}-{}".format(file_prefix, games_played))

try:
    if __name__ == '__main__':
        try:
            main()
        except:
            logging.exception("Error in main program")
            raise
except Exception as e:
    print(e)
finally:
    subprocess.call(["pkill", "fake"])
    subprocess.call(["pkill", "halite"])
    print("The end is nigh.")


