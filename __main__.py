import gymnasium as gym


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from experiment.dqn import run_dqn, dqn_log_root

from experiment.ppo import run_ppo, ppo_log_root

from env import AggregatedEnv


env_monitor_args = dict(
    filename = dqn_log_root,
)

if __name__ == "__main__":

    # env = Monitor(AggregatedEnv("CartPole-v1"), **env_monitor_args)
    # eval_env = Monitor(AggregatedEnv("CartPole-v1"), **env_monitor_args)
    
    # env = Monitor(gym.make("CartPole-v1"), **env_monitor_args)
    # eval_env = Monitor(gym.make("CartPole-v1"), **env_monitor_args)
    
    env_name = "Acrobot-v1"#"MountainCar-v0" #"CartPole-v1"
    n_envs = 10
    
    
    env = make_vec_env(lambda : gym.make(env_name, max_episode_steps=400), n_envs = n_envs)
    eval_env = make_vec_env(lambda : gym.make(env_name,  max_episode_steps=400), n_envs = n_envs)
    
    #env = make_vec_env(lambda : AggregatedEnv(env_name), n_envs = n_envs)
    #eval_env = make_vec_env(lambda : AggregatedEnv(env_name), n_envs = n_envs)
    
    results = run_dqn(env, eval_env)
    
    print(results)


    
# 需要考虑序列模型如何接入这个库的问题

# https://rl-baselines3-zoo.readthedocs.io/   hyperparameters/algo_name.yml 使用这些超参数重构训练代码