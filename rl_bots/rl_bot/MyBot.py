from rl_bot import net as anet

import hlt
import logging
import math, random
import numpy
import torch
import platform

NUM_FEATURES = 7
NUM_OUTPUT_FEATURES = 3
HAS_CUDA = torch.cuda.is_available() and (platform.system() != 'Windows')
logging.info((platform.system()))

def convert_map_to_tensor(game_map, input_tensor, my_ships):
    my_ships.clear()

    # feature vector: [ship hp, ship friendliness, docking status, planet hp, planet size, % docked_ships, planet friendliness]
    for player in game_map.all_players():
        owner_feature = 0 if player.id == game_map.my_id else 1
        for ship in player.all_ships():
            x = int(ship.x)
            y = int(ship.y)
            # hp from [0, 1]
            input_tensor[0][0][x][y] = ship.health / 255.0
            # friendless: 0 if me, 1 if enemy
            input_tensor[0][1][x][y] = owner_feature
            # 0 if undocked, .33 if docked, .66 if docking, 1 if undocking
            input_tensor[0][2][x][y] = ship.docking_status.value / 3.0
            if owner_feature == 0:
                my_ships[(x, y)] = ship

    for planet in game_map.all_planets():
        x = int(planet.x)
        y = int(planet.y)
        # hp from [0, 1]
        input_tensor[0][3][x][y] = planet.health / (planet.radius * 255.0)
        # radius from [0, 1]
        input_tensor[0][4][x][y] = (planet.radius - 3) / 5.0
        # % of docked ships [0, 1]
        input_tensor[0][5][x][y] = len(planet.all_docked_ships()) / planet.num_docking_spots
        # owner of this planet: -1 if me, 1 if enemy, 0 if unowned
        input_tensor[0][6][x][y] = (-1 if planet.owner == game_map.my_id else 1) if planet.is_owned() else 0

def one_or_negative_one():
    return 1 if random.random() > .5 else -1

def distribution():
    return (1 - math.sqrt(1 - random.random()))

def main():
    # GAME START
    game = hlt.Game("Anathema")
    logging.info("Starting << anathema >>")

    # Initialize zeroed input tensor
    input_tensor = torch.FloatTensor(1, NUM_FEATURES, game.map.width, game.map.height).zero_()
    output_tensor = torch.FloatTensor(1, NUM_OUTPUT_FEATURES, game.map.width, game.map.height).zero_()

    if HAS_CUDA:
        input_tensor = input_tensor.cuda()
        output_tensor = output_tensor.cuda()
        logging.info("Made it here")

    net = anet.Net()
    outputs = []
    my_ships = {}

    while True:
        # TURN START
        game_map = game.update_map()
        command_queue = []

        # Rebuild our input tensor based on the map state for this turn
        convert_map_to_tensor(game_map, input_tensor, my_ships)
        vi = torch.autograd.Variable(input_tensor)

        if HAS_CUDA:
            vi = vi.cuda()

        move_commands = net.forward(vi)[0].permute(1, 2, 0)

        for (x, y) in my_ships:
            this_ship = my_ships[(x, y)]
            angle, speed, dock = move_commands[x][y].data

            angle = (angle + (one_or_negative_one() * distribution()))
            output_tensor[0][0][x][y] = angle
            command_angle = int(360 * angle) % 360

            speed = speed + (one_or_negative_one() * distribution())
            output_tensor[0][1][x][y] = speed
            command_speed = numpy.clip(int(7 * speed), 0, 7)

            dock = dock + (one_or_negative_one() * distribution())
            output_tensor[0][2][x][y] = dock
            command_dock = dock

            outputs.append(output_tensor)

            # Execute ship command
            if command_dock < .5:
                # we want to undock
                if this_ship.docking_status.value == this_ship.DockingStatus.DOCKED:
                    command_queue.append(this_ship.undock())
                else:
                    command_queue.append(this_ship.thrust(command_speed, command_angle))
            else:
                # we want to dock
                if this_ship.docking_status.value == this_ship.DockingStatus.UNDOCKED:
                    closest_planet = this_ship.closest_planet(game_map)
                    if this_ship.can_dock(closest_planet):
                        command_queue.append(this_ship.dock(closest_planet))
                else:
                    command_queue.append(this_ship.thrust(command_speed, command_angle))

        # Send our set of commands to the Halite engine for this turn
        game.send_command_queue(command_queue)
        # TURN END
    
    # GAME END


if __name__ == '__main__':
    try:
        main()
    except:
        logging.exception("Error in main program")
        raise
