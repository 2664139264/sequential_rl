# import gymnasium as gym


# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.vec_env import VecMonitor
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_checker import check_env

# from experiment.dqn import run_dqn, dqn_log_root

# from experiment.ppo import run_ppo, ppo_log_root

# from env import AggregatedEnv


# env_monitor_args = dict(
#     filename = dqn_log_root,
# )

# if __name__ == "__main__":

#     # env = Monitor(AggregatedEnv("CartPole-v1"), **env_monitor_args)
#     # eval_env = Monitor(AggregatedEnv("CartPole-v1"), **env_monitor_args)
    
#     # env = Monitor(gym.make("CartPole-v1"), **env_monitor_args)
#     # eval_env = Monitor(gym.make("CartPole-v1"), **env_monitor_args)
    
#     env = make_vec_env("CartPole-v1", n_envs = 20)
#     eval_env = make_vec_env("CartPole-v1", n_envs = 20)
    
#     results = run_dqn(env, eval_env)
    
#     print(results)


from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__=="__main__":
    env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, device="cpu", verbose=1)
    model.learn(total_timesteps=25_000)
    
    
# 需要考虑序列模型如何接入这个库的问题
# 需要看那篇文章
# 需要调查为什么vecenv中dqn算法的 cartpole episode length 表现奇怪