#!/bin/bash

COMMIT_INFO=""
read -t 100 -p "Input Commit Info: " COMMIT_INFO

for DIR in ./stable-baselines3 ../stable-baselines-contrib ../rl-baselines3-zoo ..
do
    cd $DIR
    echo "Commit $DIR"
    git commit -m $COMMIT_INFO
    git push
done