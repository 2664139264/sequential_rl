#!/bin/bash

for (( i=2 ; i<11 ; i += 2))
do

    echo "Start training CartPoleExponentiated$i\_10 using PPO"
    python -m rl_zoo3.train --algo ppo --env CartPoleExponentiated$i\_10-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/ExpCartPolePPO$i.txt
    echo "Start training CartPoleExponentiated$i\_10 using PPOLSTM"
    python -m rl_zoo3.train --algo ppo_lstm --env CartPoleExponentiated$i\_10-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/ExpCartPolePPOLSTM$i.txt

    echo "Start training CartPoleExpInv$i\_10 using PPO"
    python -m rl_zoo3.train --algo ppo --env CartPoleExpInv$i\_10-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/InvExpCartPolePPO$i.txt
    echo "Start training CartPoleExpInv$i\_10 using PPOLSTM"
    python -m rl_zoo3.train --algo ppo_lstm --env CartPoleExpInv$i\_10-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/InvExpCartPolePPOLSTM$i.txt



    echo "Start training PendulumExponentiated$i\_10 using PPO"
    python -m rl_zoo3.train --algo ppo --env PendulumExponentiated$i\_10-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/ExpPendulumPPO$i.txt
    echo "Start training PendulumExponentiated$i\_10 using PPOLSTM"
    python -m rl_zoo3.train --algo ppo_lstm --env PendulumExponentiated$i\_10-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/ExpPendulumPPOLSTM$i.txt


    echo "Start training PendulumExpInv$i\_10 using PPO"
    python -m rl_zoo3.train --algo ppo --env PendulumExpInv$i\_10-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/InvExpPendulumPPO$i.txt
    echo "Start training PendulumExpInv$i\_10 using PPOLSTM"
    python -m rl_zoo3.train --algo ppo_lstm --env PendulumExpInv$i\_10-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/InvExpPendulumPPOLSTM$i.txt




    # echo "Start training LunarLanderExponentiated$i\_10 using PPO"
    # python -m rl_zoo3.train --algo ppo --env LunarLanderExponentiated$i\_10-v2 --eval-freq 20000 --save-freq 500000 > ./experiment_log/ExpLunarLanderPPO$i.txt
    # echo "Start training LunarLanderExponentiated$i\_10 using PPOLSTM"
    # python -m rl_zoo3.train --algo ppo_lstm --env LunarLanderExponentiated$i\_10-v2 --eval-freq 20000 --save-freq 500000 > ./experiment_log/ExpLunarLanderPPOLSTM$i.txt

done