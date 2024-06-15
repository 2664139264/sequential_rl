import gymnasium as gym


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor

from experiment.dqn import run_dqn

from experiment.ppo import run_ppo, log_root
env_monitor_args = dict(
    filename = log_root,
    info_keywords = (),
)

make_lander = lambda: Monitor(gym.make("LunarLander-v2"), **env_monitor_args)

if __name__ == "__main__":
    env = make_vec_env("LunarLander-v2", n_envs = 50)
    env = (VecMonitor if isinstance(env, VecEnv) else Monitor)(env, **env_monitor_args)
    eval_env = make_vec_env("LunarLander-v2", n_envs = 5)
    eval_env = (VecMonitor if isinstance(env, VecEnv) else Monitor)(env, **env_monitor_args)

    #env = SubprocVecEnv([make_lander] * 50)
    #eval_env = make_lander()

    result = run_ppo(env, eval_env)
    print(result)
