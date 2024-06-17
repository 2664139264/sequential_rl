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
    
    env_name = "MountainCar-v0"#"Acrobot-v1"# #"CartPole-v1"
    n_envs = 5
    
    
    env = make_vec_env(lambda : gym.make(env_name, max_episode_steps=400), n_envs = n_envs)
    eval_env = make_vec_env(lambda : gym.make(env_name,  max_episode_steps=400), n_envs = n_envs)
    
    #env = make_vec_env(lambda : AggregatedEnv(env_name), n_envs = n_envs)
    #eval_env = make_vec_env(lambda : AggregatedEnv(env_name), n_envs = n_envs)
    
    results = run_ppo(
        env,
        eval_env,
        
    )
    
    print(results)


    
# 需要考虑序列模型如何接入这个库的问题

# https://rl-baselines3-zoo.readthedocs.io/   hyperparameters/algo_name.yml 使用这些超参数重构训练代码




"""
atari:
  env_wrapper:
    - stable_baselines3.common.atari_wrappers.AtariWrapper
  frame_stack: 4
  policy: 'CnnPolicy'
  n_timesteps: !!float 1e7
  buffer_size: 100000
  learning_rate: !!float 1e-4
  batch_size: 32
  learning_starts: 100000
  target_update_interval: 1000
  train_freq: 4
  gradient_steps: 1
  exploration_fraction: 0.1
  exploration_final_eps: 0.01
  # If True, you need to deactivate handle_timeout_termination
  # in the replay_buffer_kwargs
  optimize_memory_usage: False

# Almost Tuned
CartPole-v1:
  n_timesteps: !!float 5e4
  policy: 'MlpPolicy'
  learning_rate: !!float 2.3e-3
  batch_size: 64
  buffer_size: 100000
  learning_starts: 1000
  gamma: 0.99
  target_update_interval: 10
  train_freq: 256
  gradient_steps: 128
  exploration_fraction: 0.16
  exploration_final_eps: 0.04
  policy_kwargs: "dict(net_arch=[256, 256])"

# Tuned
MountainCar-v0:
  n_timesteps: !!float 1.2e5
  policy: 'MlpPolicy'
  learning_rate: !!float 4e-3
  batch_size: 128
  buffer_size: 10000
  learning_starts: 1000
  gamma: 0.98
  target_update_interval: 600
  train_freq: 16
  gradient_steps: 8
  exploration_fraction: 0.2
  exploration_final_eps: 0.07
  policy_kwargs: "dict(net_arch=[256, 256])"

# Tuned
LunarLander-v2:
  n_timesteps: !!float 1e5
  policy: 'MlpPolicy'
  learning_rate: !!float 6.3e-4
  batch_size: 128
  buffer_size: 50000
  learning_starts: 0
  gamma: 0.99
  target_update_interval: 250
  train_freq: 4
  gradient_steps: -1
  exploration_fraction: 0.12
  exploration_final_eps: 0.1
  policy_kwargs: "dict(net_arch=[256, 256])"

# Tuned
Acrobot-v1:
  n_timesteps: !!float 1e5
  policy: 'MlpPolicy'
  learning_rate: !!float 6.3e-4
  batch_size: 128
  buffer_size: 50000
  learning_starts: 0
  gamma: 0.99
  target_update_interval: 250
  train_freq: 4
  gradient_steps: -1
  exploration_fraction: 0.12
  exploration_final_eps: 0.1
  policy_kwargs: "dict(net_arch=[256, 256])"


"""