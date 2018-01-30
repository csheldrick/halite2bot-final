import sys
import logging
import copy
import time

from . import game_map


class Game:
    """
    :ivar map: Current map representation
    :ivar initial_map: The initial version of the map before game starts
    """
    def _send_string(self, s):
        """
        Send data to the game. Call :function:`done_sending` once finished.

        :param str s: String to send
        :return: nothing
        """
        if hasattr(self, "to_halite_stream"):
            self.to_halite_stream.write(s)
        else:
            sys.stdout.write(s)

    def _done_sending(self):
        """
        Finish sending commands to the game.

        :return: nothing
        """
        if hasattr(self, "to_halite_stream"):
            self.to_halite_stream.write('\n')
            self.to_halite_stream.flush()
        else:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _get_string(self):
        """
        Read input from the game.

        :return: The input read from the Halite engine
        :rtype: str
        """
        if hasattr(self, "from_halite_stream"):
            result = self.from_halite_stream.readline().rstrip("\n")
            if result == 'Done.':
                print('Finished with the game.')
                raise ValueError
        else:
            result = sys.stdin.readline().rstrip('\n')

        return result

    def send_command_queue(self, command_queue):
        """
        Issue the given list of commands.

        :param list[str] command_queue: List of commands to send the Halite engine
        :return: nothing
        """
        for command in command_queue:
            self._send_string(command)

        self._done_sending()

    @staticmethod
    def _set_up_logging(tag, name):
        """
        Set up and truncate the log

        :param tag: The user tag (used for naming the log)
        :param name: The bot name (used for naming the log)
        :return: nothing
        """
        log_file = "{}_{}.log".format(tag, name)
        logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w')
        logging.info("Initialized bot {}".format(name))

    def __init__(self, name, from_halite_stream=None, to_halite_stream=None):
        """
        Initialize the bot with the given name.

        :param name: The name of the bot.
        """
        if from_halite_stream is not None:
            self.from_halite_stream = from_halite_stream
        if to_halite_stream is not None:
            self.to_halite_stream = to_halite_stream

        tag = int(self._get_string())
        Game._set_up_logging(tag, name)
        width, height = [int(x) for x in self._get_string().strip().split()]

        self._send_string(name)
        self._done_sending()
        self.map = game_map.Map(tag, width, height)
        self.update_map()
        self.initial_map = copy.deepcopy(self.map)

    def update_map(self):
        """
        Parse the map given by the engine.
        
        :return: new parsed map
        :rtype: game_map.Map
        """
        import logging
        logging.info("---NEW TURN---")
        self.map._parse(self._get_string())
        return self.map
