import numpy as np
import hlt
import heapq
from hlt.entity import Entity
try:
    from model import KerasModel
except:
    from .model import KerasModel
from math import degrees, hypot, sqrt, exp
import random, subprocess, os, time, sys

UNDOCKED = 0
DOCKING = 1
DOCKED = 2
UNDOCKING = 3
PER_PLANET_FEATURES = 11
PLANET_MAX_NUM = 28


def distance2(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def distance(x1, y1, x2, y2):
    return sqrt(distance2(x1, y1, x2, y2))


class Bot:
    def __init__(self, name):
        if "-" in name:
            self.name = name.split("-")[0]
        else:
            self.name = name
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_name = 'weights-{}.hdf5'.format(self.name.lower())
        model_location = os.path.join(current_directory, model_name)
        self._kmodel = KerasModel(name=self.name.lower(), load_weights=True, location=model_location)
        random_input_data = np.random.rand(PLANET_MAX_NUM, PER_PLANET_FEATURES)
        predictions = self._kmodel.predict(random_input_data)
        self._turns = 0
        assert len(predictions) == PLANET_MAX_NUM

    def play(self):
        game = hlt.Game(self.name)
        turns = 0
        while True:
            turns += 1
            start = time.time()
            game_map = game.update_map()
            features = self.produce_features(game_map)
            predictions = self._kmodel.predict(features)
            assignments = self.produce_ships_to_planets_assignment(game_map, predictions)
            cmd_queue = self.play_game(game_map, assignments, turns, 0, [], start, training=False)
            # cmd_queue = self.play_game(game_map, turns, 0, [], start, training=False)
            game.send_command_queue(cmd_queue)

    def produce_features(self, game_map):
        """
        For each planet produce a set of features that we will feed to the neural net. We always return an array
        with PLANET_MAX_NUM rows - if planet is not present in the game, we set all featurse to 0.

        :param game_map: game map
        :return: 2-D array where i-th row represents set of features of the i-th planet
        """
        feature_matrix = [[0 for _ in range(PER_PLANET_FEATURES)] for _ in range(PLANET_MAX_NUM)]
        all_planets = (p for p in game_map.all_planets())
        for planet in all_planets:

            # Compute "ownership" feature - 0 if planet is not occupied, 1 if occupied by us, -1 if occupied by enemy.
            if planet.owner == game_map.get_me():
                ownership = 1
            elif planet.owner is None:
                ownership = 0
            else:  # owned by enemy
                ownership = -1

            my_best_distance = 10000
            enemy_best_distance = 10000

            gravity = 0

            health_weighted_ship_distance = 0
            sum_of_health = 0
            all_players = (p for p in game_map.all_players())
            me = game_map.get_me()
            for player in all_players:
                for ship in player.all_ships():
                    d = ship.calculate_distance_between(planet)
                    if player == me:
                        my_best_distance = min(my_best_distance, d)
                        sum_of_health += ship.health
                        health_weighted_ship_distance += d * ship.health
                        gravity += ship.health / (d * d)
                    else:
                        enemy_best_distance = min(enemy_best_distance, d)
                        gravity -= ship.health / (d * d)

            distance_from_center = distance(planet.x, planet.y, game_map.width / 2, game_map.height / 2)

            health_weighted_ship_distance = health_weighted_ship_distance / sum_of_health

            remaining_docking_spots = planet.num_docking_spots - len(planet.all_docked_ships())
            signed_current_production = planet.current_production * ownership

            is_active = remaining_docking_spots > 0 or ownership != 1

            feature_matrix[planet.id] = [
                planet.health,
                remaining_docking_spots,
                planet.remaining_resources,
                signed_current_production,
                gravity,
                my_best_distance,
                enemy_best_distance,
                ownership,
                distance_from_center,
                health_weighted_ship_distance,
                is_active
            ]

        return feature_matrix

    def produce_ships_to_planets_assignment(self, game_map, predictions):
        """
        Given the predictions from the neural net, create assignment (undocked ship -> planet) deciding which
        planet each ship should go to. Note that we already know how many ships is going to each planet
        (from the neural net), we just don't know which ones.

        :param game_map: game map
        :param predictions: probability distribution describing where the ships should be sent
        :return: list of pairs (ship, planet)
        """
        undocked_ships = list(
            filter(lambda ship: ship.docking_status == ship.DockingStatus.UNDOCKED, game_map.get_me().all_ships()))
        # largest_planet = max(planet.radius for planet in game_map.all_planets())
        # greedy assignment
        assignment = []
        number_of_ships_to_assign = len(undocked_ships)
        if number_of_ships_to_assign == 0:
            return assignment
        planet_heap = []
        ship_heaps = [[] for _ in range(PLANET_MAX_NUM)]
        # Create heaps for greedy ship assignment.
        all_planets = (p for p in game_map.all_planets())
        me = game_map.get_me()
        for planet in all_planets:
            # We insert negative number of ships as a key, since we want max heap here.
            heapq.heappush(planet_heap, (-predictions[planet.id] * number_of_ships_to_assign, planet.id))
            h = []
            for ship in undocked_ships:
                d = ship.calculate_distance_between(planet)
                heapq.heappush(h, (d, ship.id))
            ship_heaps[planet.id] = h
        # Create greedy assignment
        already_assigned_ships = set()
        while number_of_ships_to_assign > len(already_assigned_ships):
            # Remove the best planet from the heap and put it back in with adjustment.
            # (Account for the fact the distribution values are stored as negative numbers on the heap.)
            ships_to_send, best_planet_id = heapq.heappop(planet_heap)
            ships_to_send = -(-ships_to_send - 1)
            heapq.heappush(planet_heap, (ships_to_send, best_planet_id))
            # Find the closest unused ship to the best planet.
            _, best_ship_id = heapq.heappop(ship_heaps[best_planet_id])
            while best_ship_id in already_assigned_ships:
                _, best_ship_id = heapq.heappop(ship_heaps[best_planet_id])
            # Assign the best ship to the best planet.
            assignment.append((me.get_ship(best_ship_id), game_map.get_planet(best_planet_id)))
            already_assigned_ships.add(best_ship_id)
        return assignment

    def navigate(self, game_map, start_of_round, ship, destination, speed):
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

    def check_enemies(self, ship, others, game_map):
        me = game_map.get_me()
        danger = [s for s in others if ship.calculate_distance_between(s) < 17]
        return danger

    def handle_defense(self, ship, others, game_map, cmd_q, planet=None):
        # others = list from check_enemies
        me = game_map.get_me()
        eothers = others
        if planet:
            mi = eothers.index(min(eothers, key=lambda s: distance(s.x, s.y, planet.x, planet.y)))
        else:
            mi = eothers.index(min(eothers, key=lambda s: distance(s.x, s.y, ship.x, ship.y)))
        t = eothers[mi]
        speed = 7
        status = ship.docking_status.value
        if status == UNDOCKED:
            cmd_q.append(ship.navigate(ship.closest_point_to(t, min_distance=3), game_map, speed))
        elif status == UNDOCKING or DOCKING:
            cmd_q.append("")
        elif status == DOCKED:
            nearby = others
            if len(nearby) > 0 and len(ship.planet.all_docked_ships()) > 1:
                cmd_q.append(ship.undock())
            else:
                cmd_q.append("")

    def handle_dock(self, ship, planet, game_map, command_queue):
        command_queue.append(ship.dock(planet))

    def handle_out_of_range(self, ship, planet, game_map, command_queue, angle, dist=None):
        if dist is None:
            dist = ship.calculate_distance_between(planet)
        point = ship.closest_point_to(planet, min_distance=3)
        command_queue.append(ship.navigate(point, game_map, 7))

    def play_game(self, game_map, assignments, turns, i, command_queue, start, training=False, graph=False):
        my_ships = {}
        taken_dmg = []
        for ship in game_map.get_me().all_ships():
            my_ships[int(ship.x), int(ship.y)] = ship
            if ship.health < 255:
                taken_dmg.append(ship.id)
        me = game_map.get_me()
        attack_mode = False
        speed = hlt.constants.MAX_SPEED
        # Enable Attack Mode based on ownership
        if turns > 10:
            my_planets = [p for p in game_map.all_planets() if p.owner == me]
            full_planets = []

            for planet in my_planets:
                if planet.docking_spots() == 0: full_planets.append(planet)
            num_p = len(game_map.all_planets())
            # 10 - 5 = 5 / 10 = .5 * 100 = 50 %
            # Check what % of planets are mine
            perc_p = ((num_p - len(my_planets)) / num_p) * 100
            if len(full_planets) == len(my_planets) and perc_p > 50:
                # All my planets have been filled
                attack_mode = True
        # for (x, y), this_ship in my_ships.items():
        aothers = [s for s in game_map._all_ships() if s.owner != me]
        for ship, planet in assignments:
            others = [s for s in aothers if s.id != ship.id]
            # Rush Defense
            if ship.id in taken_dmg:
                danger = self.check_enemies(ship, others, game_map)
                if len(danger) > 0:
                    self.handle_defense(ship, danger, game_map, command_queue)
                    continue
            # Check if docking/undocking
            if ship.docking_status.value == DOCKING or ship.docking_status.value == UNDOCKING:
                command_queue.append("")
                continue
            # Ship already docked
            elif ship.docking_status.value == DOCKED:
                danger = self.check_enemies(ship, others, game_map)
                if len(danger) > 0:
                    self.handle_defense(ship, danger, game_map, command_queue)
                    continue
                else:
                    command_queue.append("")
                    continue

            is_planet_friendly = not planet.is_owned() or planet.owner == me
            # Unowned or Mine
            if is_planet_friendly:
                # In range to dock and has space
                if ship.can_dock(planet) and not planet.is_full():
                    self.handle_dock(ship, planet, game_map, command_queue)
                    continue
                # Not in range or In Range and no space
                else:
                    danger = self.check_enemies(ship, others, game_map)
                    if len(danger) > 0:
                        command_queue.append(self.handle_defense(ship, others, game_map, command_queue, planet))
                    else:
                        command_queue.append(self.navigate(game_map, start, ship, ship.closest_point_to(planet, min_distance=2), speed))
                    continue
            # Enemy
            else:
                # Check for enemies
                #danger = self.check_enemies(ship, others, game_map)
                # Enemy planet attack mode or nearby enemies
                # if attack_mode or len(danger) > 0:
                weakest = planet.weakest_ship()
                command_queue.append(
                    ship.navigate(ship.closest_point_to(weakest, min_distance=2), game_map, 7))
                continue
                # else:
                #     angle = ship.calculate_angle_between(planet)
                #     dist = ship.calculate_distance_between(planet)
                #     if dist < 5:
                #         weakest = planet.weakest_ship()
                #         command_queue.append(ship.navigate(ship.closest_point_to(weakest, min_distacne=3), game_map, 7))
                #         continue
                #     else:
                #         self.handle_out_of_range(ship, planet, game_map, command_queue, angle, dist=dist)
                #         continue
        q = [c for c in command_queue if c is not None]
        return q
