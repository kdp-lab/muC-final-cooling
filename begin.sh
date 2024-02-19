#!/bin/bash

echo "Running $1"
nohup python3 $1 > output.txt &
tail -n +1 -f output.txt
