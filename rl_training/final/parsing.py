import os
import numpy as np
import pandas as pd

import json
from math import atan2, pi, sqrt

try:
    from tsmlstarterbot.common import *
except:
    from common import *


def angle(x, y):
    radians = atan2(y, x)
    if radians < 0:
        radians = radians + 2 * pi
    return round(radians / pi * 180)


def find_winner(data):
    for player, stats in data['stats'].items():
        if stats['rank'] == 1:
            return player
    return -1


def angle_dist(a1, a2):
    return (a1 - a2 + 360) % 360


def find_target_planet(bot_id, current_frame, planets, move):
    """
    Find a planet which the ship tried to go to. We try to find it by looking at the angle that the ship moved
    with and the angle between the ship and the planet.
    :param bot_id: id of bot to imitate
    :param current_frame: current frame
    :param planets: planets data
    :param move: current move to analyze
    :return: id of the planet that ship was moving towards
    """

    if move['type'] == 'dock':
        # If the move was to dock, we know the planet we wanted to move towards
        return move['planet_id']
    if move['type'] != 'thrust':
        # If the move was not "thrust" (i.e. it was "undock"), there is no angle to analyze
        return -1

    ship_angle = move['angle']
    ship_data = current_frame['ships'][bot_id][str(move['shipId'])]
    ship_x = ship_data['x']
    ship_y = ship_data['y']

    optimal_planet = -1
    optimal_angle = -1
    for planet_data in planets:
        planet_id = str(planet_data['id'])
        if planet_id not in current_frame['planets'] or current_frame['planets'][planet_id]['health'] <= 0:
            continue

        planet_x = planet_data['x']
        planet_y = planet_data['y']
        a = angle(planet_x - ship_x, planet_y - ship_y)
        # We try to find the planet with minimal angle distance
        if optimal_planet == -1 or angle_dist(ship_angle, a) < angle_dist(ship_angle, optimal_angle):
            optimal_planet = planet_id
            optimal_angle = a

    return optimal_planet


def format_data_for_training(data):
    """
    Create numpy array with planet features ready to feed to the neural net.
    :param data: parsed features
    :return: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
    """
    training_input = []
    training_output = []
    for d in data:
        if len(d) > 2:
            d = d[0]
        features, expected_output = d

        if len(expected_output.values()) == 0:
            continue

        features_matrix = []
        for planet_id in range(PLANET_MAX_NUM):
            if str(planet_id) in features:
                features_matrix.append(features[str(planet_id)])
            else:
                features_matrix.append([0] * PER_PLANET_FEATURES)

        fm = np.array(features_matrix)

        output = [0] * PLANET_MAX_NUM
        for planet_id, p in expected_output.items():
            output[int(planet_id)] = p
        result = np.array(output)

        training_input.append(fm)
        training_output.append(result)

    return np.array(training_input), np.array(training_output)


def serialize_data(data, dump_features_location):
    """
    Serialize all the features into .h5 file.

    :param data: data to serialize
    :param dump_features_location: path to .h5 file where the features should be saved
    """
    training_data_for_pandas = {
        (game_id, frame_id, planet_id): planet_features
        for game_id, frame in enumerate(data)
        for frame_id, planets in enumerate(frame)
        for planet_id, planet_features in planets[0].items()}

    training_data_to_store = pd.DataFrame.from_dict(training_data_for_pandas, orient="index")
    training_data_to_store.columns = FEATURE_NAMES
    index_names = ["game", "frame", "planet"]
    training_data_to_store.index = pd.MultiIndex.from_tuples(training_data_to_store.index, names=index_names)
    training_data_to_store.to_hdf(dump_features_location, "training_data")


def dump_data(data, dump_features_location="../data/features/shummie.dat"):
    with open(dump_features_location, 'w') as out:
        json.dump(data, out)
    out.close()
    print("Data Dumped to {}".format(dump_features_location))


def load_data(pack="all", partial=False):
    if pack == "all": 
        _pack = "combined"
        all = "./data/{}-full.dat".format(_pack)
        data = json.load(open(all, 'r'))
    flat_training_data = [frame for game in data for frame in game]
    nb_games = len(data)
    print("Data loaded, total games: {}".format(nb_games))
    return format_data_for_training(flat_training_data)


def parse_game(game_json_data):
    """
    Parse a game to compute features. This method computes PER_PLANET_FEATURES features for each planet in each frame
    of the game.

    :param game_json_data: list of json dictionaries describing a game
    :param dump_features_location: location where to serialize the features
    :return: data ready for training
    """
    print("Parsing game...")
    training_data = []
    bot_to_imitate_id = find_winner(game_json_data)
    bot_to_imitate = game_json_data['player_names'][int(bot_to_imitate_id)]
    print("Bot to imitate: {}.".format(bot_to_imitate))
    frames = game_json_data['frames']
    moves = game_json_data['moves']
    width = game_json_data['width']
    height = game_json_data['height']
    game_training_data = []
    # Ignore the last frame, no decision to be made there
    for idx in range(len(frames) - 1):

        current_moves = moves[idx]
        current_frame = frames[idx]

        if bot_to_imitate_id not in current_frame['ships'] or len(current_frame['ships'][bot_to_imitate_id]) == 0:
            continue

        planet_features = {}  # planet_id -> list of features per ship per planet
        current_planets = current_frame['planets']

        # find % allocation for all ships
        all_moving_ships = 0
        allocations = {}

        # for each planet we want to find how many ships are being moved towards it now
        for ship_id, ship_data in current_frame['ships'][bot_to_imitate_id].items():
            if ship_id in current_moves[bot_to_imitate_id][0]:
                p = find_target_planet(bot_to_imitate_id, current_frame,
                                       game_json_data['planets'],
                                       current_moves[bot_to_imitate_id][0][ship_id],
                                       )
                planet_id = int(p)
                if planet_id < 0 or planet_id >= PLANET_MAX_NUM:
                    continue

                if p not in allocations:
                    allocations[p] = 0
                allocations[p] = allocations[p] + 1
                all_moving_ships = all_moving_ships + 1

        if all_moving_ships == 0:
            continue

        # Compute what % of the ships should be sent to given planet
        for planet_id, allocated_ships in allocations.items():
            allocations[planet_id] = allocated_ships / all_moving_ships

        # Compute features
        for planet_id in range(PLANET_MAX_NUM):

            if str(planet_id) not in current_planets:
                continue
            planet_data = current_planets[str(planet_id)]

            gravity = 0
            planet_base_data = game_json_data['planets'][planet_id]
            closest_friendly_ship_distance = 10000
            closest_enemy_ship_distance = 10000

            ownership = 0
            if str(planet_data['owner']) == bot_to_imitate_id:
                ownership = 1
            elif planet_data['owner'] is not None:
                ownership = -1

            average_distance = 0
            my_ships_health = 0

            for player_id, ships in current_frame['ships'].items():
                for ship_id, ship_data in ships.items():
                    is_bot_to_imitate = 1 if player_id == bot_to_imitate_id else -1
                    dist2 = distance2(planet_base_data['x'], planet_base_data['y'], ship_data['x'], ship_data['y'])
                    dist = sqrt(dist2)
                    gravity = gravity + is_bot_to_imitate * ship_data['health'] / dist2
                    if is_bot_to_imitate == 1:
                        closest_friendly_ship_distance = min(closest_friendly_ship_distance, dist)
                        average_distance = average_distance + dist * ship_data['health']
                        my_ships_health = my_ships_health + ship_data['health']
                    else:
                        closest_enemy_ship_distance = min(closest_enemy_ship_distance, dist)

            distance_from_center = distance(planet_base_data['x'], planet_base_data['y'], width / 2, height / 2)
            average_distance = average_distance / my_ships_health

            is_active = 1.0 if planet_base_data['docking_spots'] > len(
                planet_data['docked_ships']) or ownership != 1 else 0.0

            signed_current_production = planet_data['current_production'] * ownership

            # Features of the planet are inserted into the vector in the order described by FEATURE_NAMES
            planet_features[str(planet_id)] = [
                planet_data['health'],
                planet_base_data['docking_spots'] - len(planet_data['docked_ships']),
                planet_data['remaining_production'],
                signed_current_production,
                gravity,
                closest_friendly_ship_distance,
                closest_enemy_ship_distance,
                ownership,
                distance_from_center,
                average_distance,
                is_active]

        game_training_data.append((planet_features, allocations))
    training_data.append(game_training_data)
    return training_data
    # if dump_features_location is not None:
    #     dump_data(training_data, dump_features_location)

    # flat_training_data = [item for sublist in training_data for item in sublist]
    # return format_data_for_training(flat_training_data)


def parse(all_games_json_data, bot_to_imitate=None, dump_features_location=None):
    """
    Parse the games to compute features. This method computes PER_PLANET_FEATURES features for each planet in each frame
    in each game the bot we're imitating played.

    :param all_games_json_data: list of json dictionaries describing games
    :param bot_to_imitate: name of the bot to imitate or None if we want to imitate the bot who won the most games
    :param dump_features_location: location where to serialize the features
    :return: data ready for training
    """
    print("Parsing data...")

    parsed_games = 0

    training_data = []

    # if bot_to_imitate is None:
    #     print("No bot name provided, choosing the bot with the highest number of games won...")
    #     players_games_count = {}
    #     for json_data in all_games_json_data:
    #         w = find_winner(json_data)
    #         p = json_data['player_names'][int(w)]
    #         if p not in players_games_count:
    #             players_games_count[p] = 0
    #         players_games_count[p] += 1

        # bot_to_imitate = max(players_games_count, key=players_games_count.get)
    # print("Bot to imitate: {}.".format(bot_to_imitate))
    c = 0
    for json_data in all_games_json_data:
        if bot_to_imitate is None:
            print("No bot name provided, choosing bot per match")
            player_games_count = {}
            w = find_winner(json_data)
            p = json_data['player_names'][int(w)]
            if p not in player_games_count:
                player_games_count[p] = 0
            player_games_count[p] += 1
        bot_to_imitate = max(player_games_count, key=player_games_count.get)
        c += 1
        frames = json_data['frames']
        moves = json_data['moves']
        width = json_data['width']
        height = json_data['height']

        # For each game see if bot_to_imitate played in it
        if bot_to_imitate not in set(json_data['player_names']):
            continue
        # We train on all the games of the bot regardless whether it won or not.
        bot_to_imitate_id = str(json_data['player_names'].index(bot_to_imitate))

        parsed_games = parsed_games + 1
        game_training_data = []

        # Ignore the last frame, no decision to be made there
        for idx in range(len(frames) - 1):

            current_moves = moves[idx]
            current_frame = frames[idx]

            if bot_to_imitate_id not in current_frame['ships'] or len(current_frame['ships'][bot_to_imitate_id]) == 0:
                continue

            planet_features = {}  # planet_id -> list of features per ship per planet
            current_planets = current_frame['planets']

            # find % allocation for all ships
            all_moving_ships = 0
            allocations = {}

            # for each planet we want to find how many ships are being moved towards it now
            for ship_id, ship_data in current_frame['ships'][bot_to_imitate_id].items():
                if ship_id in current_moves[bot_to_imitate_id][0]:
                    p = find_target_planet(bot_to_imitate_id, current_frame,
                                           json_data['planets'],
                                           current_moves[bot_to_imitate_id][0][ship_id],
                                           )
                    planet_id = int(p)
                    if planet_id < 0 or planet_id >= PLANET_MAX_NUM:
                        continue

                    if p not in allocations:
                        allocations[p] = 0
                    allocations[p] = allocations[p] + 1
                    all_moving_ships = all_moving_ships + 1

            if all_moving_ships == 0:
                continue

            # Compute what % of the ships should be sent to given planet
            for planet_id, allocated_ships in allocations.items():
                allocations[planet_id] = allocated_ships / all_moving_ships

            # Compute features
            for planet_id in range(PLANET_MAX_NUM):

                if str(planet_id) not in current_planets:
                    continue
                planet_data = current_planets[str(planet_id)]

                gravity = 0
                planet_base_data = json_data['planets'][planet_id]
                closest_friendly_ship_distance = 10000
                closest_enemy_ship_distance = 10000

                ownership = 0
                if str(planet_data['owner']) == bot_to_imitate_id:
                    ownership = 1
                elif planet_data['owner'] is not None:
                    ownership = -1

                average_distance = 0
                my_ships_health = 0

                for player_id, ships in current_frame['ships'].items():
                    for ship_id, ship_data in ships.items():
                        is_bot_to_imitate = 1 if player_id == bot_to_imitate_id else -1
                        dist2 = distance2(planet_base_data['x'], planet_base_data['y'], ship_data['x'], ship_data['y'])
                        dist = sqrt(dist2)
                        gravity = gravity + is_bot_to_imitate * ship_data['health'] / dist2
                        if is_bot_to_imitate == 1:
                            closest_friendly_ship_distance = min(closest_friendly_ship_distance, dist)
                            average_distance = average_distance + dist * ship_data['health']
                            my_ships_health = my_ships_health + ship_data['health']
                        else:
                            closest_enemy_ship_distance = min(closest_enemy_ship_distance, dist)

                distance_from_center = distance(planet_base_data['x'], planet_base_data['y'], width / 2, height / 2)
                average_distance = average_distance / my_ships_health

                is_active = 1.0 if planet_base_data['docking_spots'] > len(
                    planet_data['docked_ships']) or ownership != 1 else 0.0

                signed_current_production = planet_data['current_production'] * ownership

                # Features of the planet are inserted into the vector in the order described by FEATURE_NAMES
                planet_features[str(planet_id)] = [
                    planet_data['health'],
                    planet_base_data['docking_spots'] - len(planet_data['docked_ships']),
                    planet_data['remaining_production'],
                    signed_current_production,
                    gravity,
                    closest_friendly_ship_distance,
                    closest_enemy_ship_distance,
                    ownership,
                    distance_from_center,
                    average_distance,
                    is_active]

            game_training_data.append((planet_features, allocations))
        # dump_data(game_training_data, dump_features_location="./data/features/game-{}.dat".format(c))
        training_data.append(game_training_data)

    if parsed_games == 0:
        raise Exception("Didn't find any matching games. Try different bot.")

    if dump_features_location is not None:
        # serialize_data(training_data, dump_features_location)
        dump_data(training_data, dump_features_location)

    flat_training_data = [item for sublist in training_data for item in sublist]

    print("Data parsed, parsed {} games, total frames: {}".format(parsed_games, len(flat_training_data)))

    return format_data_for_training(flat_training_data)


def dumper(pack="shummie", combine=False):
    from train_keras import fetch_data_dir
    _dirs = ["./data/{}/group1/", "./data/{}/group2/", "./data/{}/group3/", "./data/{}/group4/"]
    dirs = [d.format(pack) for d in _dirs]
    if combine:
        dirs += [d.format("zxqfl") for d in _dirs]
    limit = 1000
    cnt = 5
    game_data = []
    for idx, _dir in enumerate(dirs):
        print("Fetching: {}".format(_dir))
        raw_data = fetch_data_dir(_dir, limit)
        print("Found {} games".format(len(raw_data)))
        for i, data in enumerate(raw_data):
            print("Parsing game {}".format(i+1))
            game_data += parse_game(data)
        del raw_data
        print("Finished Parsing")
    print("Dumping games")
    dest = "./data/{}-full.dat".format(pack if not combine else "combined")
    dump_data(game_data, dest)
    print("Finished Dumping")



if __name__ == '__main__':
    dumper(combine=True)
