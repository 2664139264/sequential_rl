#!/bin/bash


for (( i=0 ; i<6 ; i += 1 ))
do

    echo "Start training CartPoleAggregated$i using PPO"
    python -m rl_zoo3.train --algo ppo --env CartPoleAggregated$i-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/CartPolePPO$i.txt 
    echo "Start training CartPoleAggregated$i using PPOLSTM"
    python -m rl_zoo3.train --algo ppo_lstm --env CartPoleAggregated$i-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/CartPolePPOLSTM$i.txt 
    echo "Start training CartPoleDifferentiated$i using PPO"
    python -m rl_zoo3.train --algo ppo --env CartPoleDifferentiated$i-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/DifCartPolePPO$i.txt
    echo "Start training CartPoleDifferentiated$i using PPOLSTM"
    python -m rl_zoo3.train --algo ppo_lstm --env CartPoleDifferentiated$i-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/DifCartPolePPOLSTM$i.txt

    echo "Start training PendulumAggregated$i using PPO"
    python -m rl_zoo3.train --algo ppo --env PendulumAggregated$i-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/PendulumPPO$i.txt
    echo "Start training PendulumAggregated$i using PPOLSTM"
    python -m rl_zoo3.train --algo ppo_lstm --env PendulumAggregated$i-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/PendulumPPOLSTM$i.txt
    echo "Start training PendulumDifferentiated$i using PPO"
    python -m rl_zoo3.train --algo ppo --env PendulumDifferentiated$i-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/DifPendulumPPO$i.txt
    echo "Start training PendulumDifferentiated$i using PPOLSTM"
    python -m rl_zoo3.train --algo ppo_lstm --env PendulumDifferentiated$i-v1 --eval-freq 20000 --save-freq 500000 > ./experiment_log/DifPendulumPPOLSTM$i.txt


    # echo "Start training LunarLanderAggregated$i using PPO"
    # python -m rl_zoo3.train --algo ppo --env LunarLanderAggregated$i-v2 --eval-freq 20000 --save-freq 500000 > ./experiment_log/LunarLanderPPO$i.txt
    # echo "Start training LunarLanderAggregated$i using PPOLSTM"
    # python -m rl_zoo3.train --algo ppo_lstm --env LunarLanderAggregated$i-v2 --eval-freq 20000 --save-freq 500000 > ./experiment_log/LunarLanderPPOLSTM$i.txt
    # echo "Start training LunarLanderDifferentiated$i using PPO"
    # python -m rl_zoo3.train --algo ppo --env LunarLanderDifferentiated$i-v2 --eval-freq 20000 --save-freq 500000 > ./experiment_log/DifLunarLanderPPO$i.txt
    # echo "Start training LunarLanderDifferentiated$i using PPOLSTM"
    # python -m rl_zoo3.train --algo ppo_lstm --env LunarLanderDifferentiated$i-v2 --eval-freq 20000 --save-freq 500000 > ./experiment_log/DifLunarLanderPPOLSTM$i.txt
done

