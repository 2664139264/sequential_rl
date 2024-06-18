#!/bin/bash

python -m rl_zoo3.train --algo ppo_lstm --env CartPoleAggregated-v1 --eval-freq 10000 --save-freq 50000