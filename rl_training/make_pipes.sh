#!/bin/bash
rm -r ./gradbot/pipes
mkdir ./gradbot/pipes
mkfifo gradbot/pipes/to_halite_0
mkfifo gradbot/pipes/from_halite_0
mkfifo gradbot/pipes/to_halite_1
mkfifo gradbot/pipes/from_halite_1
