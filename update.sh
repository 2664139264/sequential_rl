#!/bin/bash

COMMIT_INFO=""
read -t 1024 -p "Input Commit Info: " COMMIT_INFO

for DIR in ./stable-baselines3 ../stable-baselines3-contrib ../rl-baselines3-zoo ..
do
    cd $DIR
    echo "Commit $DIR"
    git add .
    git commit -m $COMMIT_INFO
    git push
done