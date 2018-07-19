import numpy as np
import hlt
from hlt.entity import Ship
from math import degrees, sqrt, exp
import subprocess, os, time, sys
try:
    from utils import get_gradient, get_gx, distance, vector, euclidean_norm, plotter, timeit
except ImportError:
    from gradbot.utils import get_gradient, get_gx, distance, vector, euclidean_norm, plotter, timeit

NUM_PLAYERS = 1
NUM_GAMES = 1000

UNDOCKED = 0
DOCKING = 1
DOCKED = 2
UNDOCKING = 3

def get_moves_old(game_map, turns, pid, training=False, graph=False):
    w = game_map.width
    h = game_map.height
    me = game_map.get_me()    
    out = np.zeros((w, h))
    graph_o, all_axes, F, Z = get_gx(game_map, training)
    gradU, gradV = np.gradient(Z, axis=(0, 1))  # gradient of func
    for idx, ship in enumerate(me.all_ships()):
        sx, sy = int(ship.x), int(ship.y)
        sv = vector(ship)  # vector
        # Distance/Magnitude/Norm/Length = np.sqrt(x**2+y**2) = np.sqrt([x,y].dot([x,y])
        dx, dy = gradU[sx][sy], gradV[sx][sy]  # unit vector of grad @ sx,sy
        gm = F.norm(dx, dy)
        u, v = -gm * dx, -gm * dy
        angle = degrees(np.arctan2(v, u)) % 360
        out[sx][sy] = angle
        is_last_ship = idx == len(me.all_ships()) - 1 
        is_my_pid = pid == 0
        if is_last_ship and is_my_pid and graph: 
            plotter(Z, sv, graph_o, turns, pid, w, h)
    graph_o.clear()
    return out

def get_moves(game_map, turns, pid, training=False, graph=False):
    w = game_map.width
    h = game_map.height
    me = game_map.get_me()
    ships = me.all_ships()
    out = {}
    if graph:
        grad_u, grad_v, graph_objs, Func, gridZ = get_gradient(ships, game_map, graph)
    else:
        grad_u, grad_v = get_gradient(ships, game_map, graph)
    for idx, ship in enumerate(ships):
        sx, sy = int(ship.x), int(ship.y)
        # Distance/Magnitude/Norm/Length = np.sqrt(x**2+y**2) = np.sqrt([x,y].dot([x,y])
        sv = vector(ship)
        u, v = grad_u[idx], grad_v[idx]
        angle = degrees(np.arctan2(v, u)) % 360
        out[sx, sy] = angle
        is_last_ship = idx == len(ships) - 1
        is_my_pid = pid == 0
        if graph and is_last_ship and is_my_pid:
            plotter(gridZ, sv, graph_objs, turns, pid, w, h)
    graph_objs.clear()
    return out


def handle_dock(ship, planet, command_queue):
    command_queue.append(ship.dock(planet))


def navigate(game_map, start_of_round, ship, destination, speed):
    """
    Send a ship to its destination. Because "navigate" method in Halite API is expensive, we use that method only if
    we haven't used too much time yet.

    :param game_map: game map
    :param start_of_round: time (in seconds) between the Epoch and the start of this round
    :param ship: ship we want to send
    :param destination: destination to which we want to send the ship to
    :param speed: speed with which we would like to send the ship to its destination
    :return:
    """
    current_time = time.time()
    have_time = current_time - start_of_round < 1.2
    navigate_command = None
    if have_time:
        navigate_command = ship.navigate(destination, game_map, speed=speed, max_corrections=100, angular_step=2)
    if navigate_command is None:
        # ship.navigate may return None if it cannot find a path. In such a case we just thrust.
        dist = ship.calculate_distance_between(destination)
        speed = speed if (dist >= speed) else dist
        navigate_command = ship.thrust(speed, ship.calculate_angle_between(destination))
    return navigate_command


def check_enemies(ship, others):
    danger = [s for s in others if ship.calculate_distance_between(s) < 17]
    return danger


def handle_defense(ship, others, game_map, cmd_q, planet=None):
    # others = list from check_enemies
    eothers = others
    if planet:
        mi = eothers.index(min(eothers, key=lambda s: distance(s.x, s.y, planet.x, planet.y)))
    else:
        mi = eothers.index(min(eothers, key=lambda s: distance(s.x, s.y, ship.x, ship.y)))
    t = eothers[mi]
    speed = 7
    status = ship.docking_status.value
    if status == UNDOCKED:
        cmd_q.append(ship.navigate(ship.closest_point_to(t, min_distance=2), game_map, speed))
    elif status == UNDOCKING or DOCKING:
        cmd_q.append(ship.navigate(ship.closest_point_to(t, min_distance=2), game_map, speed))
        # per fakepsychos writeup you can issue a command on the last turn
        # of docking and undocking and have it be executed instead of
        # on the next turn
        # cmd_q.append("")
    elif status == DOCKED:
        nearby = others
        if len(nearby) > 0 and len(ship.planet.all_docked_ships()) > 1:
            cmd_q.append(ship.undock())
        else:
            cmd_q.append("")


@timeit
def play_game(game_map, turns, i, training=False, graph=False):
    command_queue = []
    my_ships = {}
    taken_dmg = []
    me = game_map.get_me()
    attack_mode = False
    for ship in me.all_ships():
        my_ships[(int(ship.x), int(ship.y))] = ship
        if ship.health < 255:
            taken_dmg.append(ship.id)
    #dockers = [s for s in my_ships.values() if s.docking_status == DOCKED or s.docking_status == DOCKING]
    #my_planets = [p for p in game_map.all_planets() if p.owner == me]
    #full_planets = [p for p in my_planets if p.docking_spots == 0]
    #num_p = len(game_map.all_planets())
    #perc_p = ((num_p - len(my_planets)) / num_p) * 100
    #if len(full_planets) == len(my_planets) and perc_p > 45:
        # All my planets are filled
    #    attack_mode = True
    #if len(dockers) == len(my_ships):
        # All ships docking or docked
    #    return command_queue

    move_commands = get_moves_old(game_map, turns, i, training, graph)
    enemy_ships = [s for s in game_map.all_ships() if s.owner != me]
    for (x, y), ship in my_ships.items():
        angle = move_commands[x, y]
        # Rush Defense
        if ship.id in taken_dmg:
            danger = check_enemies(ship, enemy_ships)
            if len(danger) > 0:
                handle_defense(ship, danger, game_map, command_queue)
                continue
        # Ship already docked
        if ship.docking_status.value == DOCKED:
            danger = check_enemies(ship, enemy_ships)
            if len(danger) > 0:
                handle_defense(ship, danger, game_map, command_queue)
                continue
            else:
                command_queue.append("")
                continue
        planet = ship.closest_planet(game_map)
        is_planet_friendly = not planet.is_owned() or planet.owner == me
        # Unowned or Mine
        if is_planet_friendly and not planet.is_full() and ship.can_dock(planet):
            # In range to dock and has space
            handle_dock(ship, planet, command_queue)
            continue
        else:
            # Enemy/Full/Neutral planet
            # Go to point in angle from moves
            point = game_map.get_point(ship, angle, 10)
            cmd = ship.navigate(point, game_map, 7)
            # cmd = ship.thrust(s, angle)
            command_queue.append(cmd)
            continue
    
    return command_queue


def run_game(num_players, graph):
    run_commands = []
    for i in range(num_players):
        run_commands.append("./fake_bot2 {}".format(i))
    if num_players == 1 or num_players == 2:
        for i in range(num_players):
            run_commands.append("python SettlerBot.py")
    w = 40 * 3
    h = 40 * 2
    # subprocess.Popen(["./halite", "-t", "-d {} {}".format(w, h)] + run_commands)
    try:
        subprocess.Popen(["./halite", "-t"] + run_commands)
    except:
        subprocess.Popen(["./halite.exe", "-t"] + run_commands)
    # GAME START
    games_per_player = []
    maps_per_player = []
    board_states_per_player = []
    outputs_per_player = []
    ships_per_player = []
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
    turns = 0
    while True:
        turns += 1
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
                        return [], []
                continue

            command_queue = play_game(game_map, turns, i, training=True, graph=graph)
            command_queue = [c for c in command_queue if c is not None]
            game.send_command_queue(command_queue)


def main():
    import argparse,os
    import manager_constants
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Halite II training")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    games_played = int(0)
    count = 0
    while True:
        count += 1
        print("Game ID:", games_played)
        for game_id in range(0, manager_constants.rollout_games):
            run_game(NUM_PLAYERS, args.graph)
            games_played += 1


if __name__ == '__main__':
    print("in main")
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        subprocess.call(["pkill", "fakebot2"])
        subprocess.call(["pkill", "halite"])
        print("The end is nigh.")
