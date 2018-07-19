import numpy as np
from scipy.interpolate import Rbf
from .gradbot import manager_constants
import hlt
from hlt.entity import Entity
from math import degrees
import random, subprocess, os

try:
    from mpl_toolkits import mplot3d
    import mpl_toolkits.mplot3d.art3d as art3d
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from matplotlib.patches import Circle
except:
    pass

NUM_PLAYERS = 1
NUM_GAMES = 1000
NUM_FEATURES = 7
NUM_OUTPUT_FEATURES = 3
UNDOCKED = 0
DOCKING = 1
DOCKED = 2
UNDOCKING = 3


class Net:
    def __init__(self, sigma=1.0):
        pass


def make_axes(length):
    return np.arange(length), np.arange(length), np.arange(length)


def make_data(items, cls='ship', nef='f', color='gray', objs=[]):
    x, y, z = np.array([]), np.array([]), np.array([])  # make_axes(len(items))
    for t in items:
        x = np.append(x, t.x)
        y = np.append(y, t.y)
        if cls is 'planet':
            r = get_radius(t, 4)
            _z = 0
            spots = t.docking_spots()
            if nef is 'f' and spots:
                _z = -spots
            elif nef is 'e' and spots:
                _z = -len(t.all_docked_ships())
            elif nef is 'u' and spots:
                _z = -spots
            z = np.append(z, t.radius)
            for l in r:
                x = np.append(x, l[0])
                y = np.append(y, l[1])
                z = np.append(z, _z)
        else:
            _z = weight(t, cls, nef)
            z = np.append(z, _z)
            r = get_radius(t, 1)
            for l in r:
                x = np.append(x, l[0])
                y = np.append(y, l[1])
                z = np.append(z, _z)
        objs.append([int(t.x), int(t.y), color, t, cls])
    return x, y, z, objs


def weight(t, cls='ship', nef='f'):
    if cls is 'ship':
        if nef is "f": return 1.5
        if nef is 'e': return -5
    elif cls is 'planet':
        if nef is 'u':
            return -2  # -3
        elif nef is 'f':
            return -2
        elif nef is 'e':
            return -2  # -2


def get_radius(p, r=4):
    r = int(p.radius) + r
    n = []
    for i in range(-r, r):
        n.append([p.x + i, p.y])
        n.append([p.x, p.y + i])
    return n

def make_edges(x,y):
    xs, ys, zs = [], [], []
    edge = 1.0
    for i in range(len(x[0])): # All cols
        xs.append(0) # first row
        ys.append(i) # each col
        zs.append(edge) # edge value
        xs.append(len(x)-1) # Last row
        ys.append(i) # each  col
        zs.append(edge) # edge value
    for i in range(len(x)): # All rows
        xs.append(i) # each row
        ys.append(0) # first col
        zs.append(edge) # edge value
        xs.append(i) # each row
        ys.append(len(x[0])-1) # Last col
        zs.append(edge)
    return xs, ys, zs

def get_gx(game_map):
    global w, h
    me = game_map.get_me()
    r = []
    fs = me.all_ships()
    gridX, gridY = np.meshgrid(np.arange(w), np.arange(h), indexing='ij')
    fsX, fsY, fsZ, o = make_data(fs, cls='ship', nef='f', color='green')
    #xs, ys, zs = [], [], []
    xs, ys, zs = make_edges(gridX,gridY)
    xs += [x for x in fsX]
    ys += [y for y in fsY]
    zs += [z for z in fsZ]
    all_planets = game_map.all_planets()
    u_planets = [p for p in all_planets if not p.is_owned()]
    if u_planets:
        uX, uY, uZ, o = make_data(u_planets, cls='planet', nef='f', objs=o)
        xs += [x for x in uX]
        ys += [y for y in uY]
        zs += [z for z in uZ]

    f_planets = [p for p in all_planets if p not in u_planets and p.owner == me]
    if f_planets:
        fX, fY, fZ, o = make_data(f_planets, cls='planet', nef='f', color='green', objs=o)
        xs += [x for x in fX]
        ys += [y for y in fY]
        zs += [z for z in fZ]

    e_planets = [p for p in all_planets if p not in u_planets and p.owner != me]
    if e_planets:
        eX, eY, eZ, o = make_data(e_planets, cls='planet', nef='e', color='red', objs=o)
        xs += [x for x in eX]
        ys += [y for y in eY]
        zs += [z for z in eZ]

    es = [s for s in game_map._all_ships() if s.owner != me]
    if es:
        esX, esY, esZ, o = make_data(es, cls='ship', nef='e', color='red', objs=o)
        xs += [x for x in esX]
        ys += [y for y in esY]
        zs += [z for z in esZ]

    r.append(o)
    allx = np.array(xs)
    ally = np.array(ys)
    allz = np.array(zs)

    def func(self, r):
        return np.exp(-(r / self.epsilon) ** 2)

    F = Rbf(allx, ally, allz, function='gaussian', smooth=-5, epsilon=17)
    r.append(F)
    G = F(gridX, gridY)
    r.append(G)
    return r


def plotter(Z, s, o, turn, id, tU, tV, p):
    global w, h
    af = plt.figure()
    size = af.get_size_inches()
    af.set_size_inches((size[0] * 2, size[1] * 2))
    ax2 = af.add_subplot(211)
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y, indexing='ij')
    U, V = np.gradient(Z, axis=(0, 1))
    sx, sy = int(s[0]), int(s[1])
    xx = X[sx, sy]
    yy = Y[sx, sy]
    uu = -U[sx, sy]
    vv = -V[sx, sy]
    zz = Z[sx, sy]
    ships = [s for s in o if s[-1] == 'ship']
    planets = [s for s in o if s[-1] == 'planet']
    items = ships  # + planets
    Xpts, Ypts, Upts, Vpts, Zpts = np.zeros(len(items)), np.zeros(len(items)), np.zeros(len(items)), np.zeros(
        len(items)), np.zeros(len(items))
    for i, l in enumerate(items):
        Xpts[i] = l[0]
        Ypts[i] = l[1]
        Upts[i] = -U[l[0], l[1]]
        Vpts[i] = -V[l[0], l[1]]
        Zpts[i] = Z[l[0], l[1]]
    skip = (slice(None, None, 3), slice(None, None, 3))
    ax3 = af.add_subplot(212, projection='3d')
    ax3.set_xlim(0, w)
    ax3.set_ylim(0, h)
    ax3.set_zlim(-5, 5)
    ax3.scatter(Xpts, Ypts, Zpts, color='purple', alpha=1)
    ax3.plot_surface(X, Y, Z, cmap='terrain')
    ax2.set_xlim(0, w)
    ax2.set_ylim(0, h)
    ax2.contour(X, Y, Z, cmap='gist_heat')
    # ax2.streamplot(X.T,Y.T,-U.T,-V.T,cmap="magma")
    # ax2.quiver(Xpts,Ypts,Upts,Vpts, cmap='gist_heat', pivot='t	ail',alpha=.5)
    # ax2.quiver(X[skip],Y[skip],-U[skip],-V[skip],Z[skip],alpha=.5,cmap='gist_heat', pivot='tail',angles='xy',scale_units='xy')
    for l in o:
        if sx == l[0] and sy == l[1]:
            art = Circle((sx, sy), l[3].radius, color='blue')
            ax2.add_artist(art)
            ax2.text(l[0], l[1], l[3].id, color='blue')
        else:
            if l[-1] == 'planet':
                art = Circle((l[0], l[1]), l[3].radius + 2, color='yellow', alpha=.5)
                ax2.add_artist(art)
                art = Circle((l[0], l[1]), l[3].radius, color=l[2], alpha=.5)
                ax2.add_artist(art)
                ax2.text(l[0], l[1], l[3].docking_spots())
            else:
                art = Circle((l[0], l[1]), l[3].radius, color=l[2], alpha=.5)
                ax2.add_artist(art)
                ax2.text(l[0], l[1], l[3].id, color=l[2] if l[2] is 'red' else 'blue')

    plt.savefig("grads/last-{}.png".format(p))
    plt.clf()
    plt.close()


def vector(x):
    return np.array([x.x, x.y])


def unit_vector(x):
    return x / np.linalg.norm(x)


def get_moves(game_map, net, my_ships, turns, pid):
    global w, h
    my_ships.clear()
    me = game_map.get_me()
    for i, s in enumerate(me.all_ships()):
        my_ships[(int(s.x), int(s.y))] = s
    out = np.zeros((w, h))
    # f_ships, e_ships, u_planets, e_planet, f_planets,
    o,F, G = get_gx(game_map)
    for idx, ship in enumerate(me.all_ships()):
        sx, sy = int(ship.x), int(ship.y)
        u = vector(ship)  # vector
        # Distance/Magnitude/Norm/Length = np.sqrt(x**2+y**2) = np.sqrt([x,y].dot([x,y])
        gradU, gradV = np.gradient(G, axis=(0, 1))  # gradient of func
        dx, dy = gradU[sx][sy], gradV[sx][sy]  # unit vector of grad @ sx,sy
        #dxy = np.array([gradU[sx][sy], gradV[sx][sy]])  # unit vector of grad @ sx,sy
        gm = F.norm(dx, dy)
        U, V = -gm * dx, -gm * dy
        A = degrees(np.arctan2(V, U)) % 360
        out[sx][sy] = A
        if idx == len(me.all_ships()) - 1 and pid == 0: plotter(G, u, o, turns, ship.id, U, V, pid)
    o.clear()
    return out


def check_enemies(ship, others, game_map):
    #checkx = range(-15 + int(ship.x), 15 + int(ship.x))
    #checky = range(-15 + int(ship.y), 15 + int(ship.y))
    me = game_map.get_me()
    eothers = [s for s in others if s.owner != me]
    danger =  [s for s  in eothers if ship.calculate_distance_between(s) < 15]
    #danger = [s for s in eothers if int(s.x) in checkx and int(s.y) in checky]
    return danger


def handle_defense(ship, others, game_map, cmd_q, planet=None):
    # others = list from check_enemies
    me = game_map.get_me()
    eothers = others
    if planet:
        mi = eothers.index(min(eothers, key=lambda s: np.sqrt((s.x - planet.x) ** 2 + (s.y - planet.y) ** 2)))
    else:
        mi = eothers.index(min(eothers, key=lambda s: np.sqrt((s.x - ship.x) ** 2 + (s.y - ship.y) ** 2)))
    t = eothers[mi]
    speed = 7
    status = ship.docking_status.value
    if status == UNDOCKED:
        cmd_q.append(ship.navigate(ship.closest_point_to(t, min_distance=1), game_map, speed))
    elif status == UNDOCKING or DOCKING:
        cmd_q.append("")
    elif status == DOCKED:
        nearby = others #[eothers[i] for i, d in eothers if np.sqrt((d.x - ship.x) ** 2 + (d.y - ship.y) ** 2) < 15 and i != mi]
        if len(nearby) > 0 and len(ship.planet.all_docked_ships()) > 1:
            cmd_q.append(ship.undock())
        else:
            cmd_q.append("")


def handle_dock(ship, planet, game_map, command_queue):
    command_queue.append(ship.dock(planet))


def get_speed(ship, target, type='planet'):
    dist = int(ship.calculate_distance_between(target))
    dist -= target.radius
    if type is 'planet':
        dist -= 2  # Docking radius = 4
    return 7 if dist > 7 else dist if dist > 0 else 0


def run_game(num_players, net):
    run_commands = []
    for i in range(num_players):
        run_commands.append("./fake_bot2 {}".format(i))
    if num_players == 1 or num_players == 2:
        for i in range(num_players):
            #c = random.choice([0, 1])
            #if c == 0:
            run_commands.append("python MyBot.py")
            #else:
            #run_commands.append("python ../SettlerBot/MyBot.py")
    global w
    global h
    w = 80 * 3
    h = 80 * 2

    subprocess.Popen(["./halite", "-d", "{} {}".format(w, h)] + run_commands)
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
    output_history = [[] for _ in range(num_players)]
    input_history = [[] for _ in range(num_players)]
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
            command_queue = []
            my_ships = ships_per_player[i]
            if len(my_ships.keys()) > 3:
                made_ships[i] = True
            move_commands = get_moves(game_map, net, my_ships, turns, i)
            me = game_map.get_me()
            attack_mode = False
            if turns > 10:
                dockers = [s for s in my_ships.values() if s.docking_status == DOCKED or s.docking_status == DOCKING]
                my_planets = [p for p in game_map.all_planets() if p.owner == me]
                full_planets = []
                for planet in my_planets:
                     if planet.docking_spots == 0: full_planets.append(planet)
                num_p = len(game_map.all_planets())
                # 10 - 5 = 5 / 10 = .5 * 100 = 50 %
                # Check what % of planets are mine
                perc_p = ((num_p - len(my_planets)) / num_p) * 100
                if len(full_planets) == len(my_planets) and perc_p > 45:
                    # All my planets have been filled
                    attack_mode = True
                if len(dockers) == len(my_ships):
                    # All docked or docking
                    game.send_command_queue(command_queue)
                    continue
            for (x, y), this_ship in my_ships.items():
                angle = move_commands[x, y]
                closest_planet = this_ship.closest_planet(game_map)
                if turns < 10:
                    point = game_map.get_point(this_ship, angle, 10)
                    command_queue.append(this_ship.navigate(point,game_map, 7))
                    continue
                others = [s for s in game_map._all_ships() if s.id != this_ship.id]
                # Check if docking/undocking
                if this_ship.docking_status.value == DOCKING or this_ship.docking_status.value == UNDOCKING:
                    command_queue.append("")
                    continue
                # Ship already docked
                elif this_ship.docking_status.value == DOCKED:
                    danger = check_enemies(this_ship, others, game_map)
                    if len(danger) > 0:
                        handle_defense(this_ship, danger, game_map, command_queue)
                        continue
                    else:
                        command_queue.append("")
                        continue
                # Check for enemies
                danger = check_enemies(this_ship, others, game_map)
                # Planet has owner
                if closest_planet.is_owned():
                    # Mine
                    if closest_planet.owner == me:
                        # Has space
                        if not closest_planet.is_full():
                            # In range to dock and no nearby enemies
                            if this_ship.can_dock(closest_planet) and len(danger) == 0:
                                # Dock
                                handle_dock(this_ship, closest_planet, game_map, command_queue)
                                continue
                            # In range and enemies nearby
                            elif this_ship.can_dock(closest_planet) and len(danger) > 0:
                                handle_defense(this_ship, danger, game_map, command_queue, planet=closest_planet)
                                continue
                            # Not in range
                            else:
                                point = game_map.get_point(this_ship, angle, 10)
                                command_queue.append(this_ship.navigate(point, game_map, 7))
                                continue
                        # No space
                        else:
                            point = game_map.get_point(this_ship, angle, 10)
                            command_queue.append(this_ship.navigate(point, game_map, 7))
                            continue
                    # Enemy
                    else:
                        # Enemy planet has no spots and attack mode or nearby enemies
                        if closest_planet.is_full() and (attack_mode or len(danger) > 0):
                            weakest = closest_planet.weakest_ship()
                            command_queue.append(this_ship.navigate(this_ship.closest_point_to(weakest, min_distance=1), game_map, 7))
                            continue
                        # Planet is full and not in attack mode and no nearby enemies? Can't happen, since they own it there has to be at least 1
                        elif closest_planet.is_full():
                            point = game_map.get_point(this_ship, angle, 10)
                            command_queue.append(this_ship.navigate(point, game_map, 7))
                            continue
                        # Planet not filled
                        else:
                            weakest = closest_planet.weakest_ship()
                            command_queue.append(this_ship.navigate(this_ship.closest_point_to(weakest, min_distance=1), game_map, 7))
                            continue
                # Planet not owned
                else:
                    # In range to dock and ship undocked
                    if this_ship.can_dock(closest_planet):
                        done = False
                        # Check for enemies
                        if attack_mode and len(danger) > 0:
                            # Nearby Enemies - handle defense
                            handle_defense(this_ship, danger, game_map, command_queue, planet=closest_planet)
                            continue
                        # In range and no enemies
                        else:
                            handle_dock(this_ship, closest_planet, game_map, command_queue)
                            continue
                    # Not in range to dock
                    else:
                        # Navigate to point
                        point = game_map.get_point(this_ship, angle, 10)
                        command_queue.append(this_ship.navigate(point, game_map, 7))
                        continue
                # Should no longer make it here
                id = this_ship.id
                status = str(this_ship.docking_status).split(".")[-1]
                p = getattr(this_ship.planet, "id") if hasattr(this_ship, "planet") and this_ship.planet else "None"
                if i == 0: print("id: {0} a: {1:.3f} status: {2} planet: {3}".format(id, angle, status, p))
            command_queue = [c for c in command_queue if c is not None]
            game.send_command_queue(command_queue)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Halite II training")
    parser.add_argument("--model_name", help="Name of the model", default="conv")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    net = Net()
    if args.load:
        net.model.load_weights("models/{}".format(args.model_name))
    file_prefix, games_played = args.model_name.split("-")
    games_played = int(games_played)
    count = 0
    while True:
        count += 1
        print("Game ID:", games_played)
        for game_id in range(0, manager_constants.rollout_games):
            run_game(NUM_PLAYERS, net)
            #subprocess.Popen(["pkill", "fakebot2"])
            #subprocess.Popen(["pkill", "halite"])
            # subprocess.Popen(["pkill", "-f", "MyBot.py"])

            games_played += 1

try:
    if __name__ == '__main__':
        try:
            main()
        except:
            print("Error in main program")
            raise
except Exception as e:
    print(e)
finally:
    subprocess.call(["pkill", "fakebot2"])
    subprocess.call(["pkill", "halite"])
    print("The end is nigh.")
