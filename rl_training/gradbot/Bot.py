from .grads import play_game
from . import hlt
import time


class Bot:
    def __init__(self, name):
        self.name = name

    def play(self):
        game = hlt.Game(self.name)
        turns = 0
        while True:
            turns += 1
            start = time.time()
            game_map = game.update_map()
            cmd_queue = play_game(game_map, turns, 0, [], start, training=False)
            game.send_command_queue(cmd_queue)
