import os
from typing import Callable, Dict, Any
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from utils import AlgoT, EnvT, PolicyInstanceT, EnvInstanceT

dqn_log_root = "./log/dqn"
dqn_algo_args = dict(
    verbose = 1,
    tensorboard_log = dqn_log_root
)
dqn_learn_args = dict(
    total_timesteps = int(2e5),
    progress_bar = True
)
dqn_eval_args = dict(
    n_eval_episodes = 10
)
dqn_eval_callback_args = dict(
    best_model_save_path = os.path.join(dqn_log_root, "ckpt"),
    log_path = os.path.join(dqn_log_root, "eval_log"),
    eval_freq = 500,
    deterministic = True,
    render = False
)

default_policy_constructor = lambda _, __: "MlpPolicy"


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
def run_dqn(env: EnvInstanceT, eval_env: EnvInstanceT,
        policy_constructor: Callable[[AlgoT, EnvT], PolicyInstanceT] = default_policy_constructor,
        algo_args: Dict[str, Any] = dqn_algo_args,
        learn_args: Dict[str, Any] = dqn_learn_args,
        eval_args: Dict[str, Any] = dqn_eval_args,
        eval_callback_args: Dict[str, Any] = dqn_eval_callback_args):

    policy = policy_constructor(DQN, env.__class__)
    model = DQN(
        policy, env, 
        **algo_args,
    )
    model.learn(
        **learn_args,
        callback = CallbackList([
            EvalCallback(
                eval_env,
                **eval_callback_args
            )
        ])
    )
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), **eval_args)
    return mean_reward, std_reward
