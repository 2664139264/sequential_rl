import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor

from experiment.dqn import run_dqn, log_root


if __name__ == "__main__":
    env = make_vec_env("LunarLander-v2", n_envs = 5)
    env_monitor_args = dict(
        filename = log_root,
        info_keywords = (),
    )
    env = (VecMonitor if isinstance(env, VecEnv) else Monitor)(env, **env_monitor_args)
    eval_env = make_vec_env("LunarLander-v2", n_envs = 5)
    eval_env = (VecMonitor if isinstance(env, VecEnv) else Monitor)(env, **env_monitor_args)
    result = run_dqn(env, eval_env)
    print(result)