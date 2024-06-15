import os
from typing import Callable, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from utils import AlgoT, EnvT, PolicyInstanceT, EnvInstanceT

ppo_log_root = "./log/ppo"
ppo_algo_args = dict(
    verbose = 1,
    tensorboard_log = ppo_log_root
)
ppo_learn_args = dict(
    total_timesteps = int(1e5),
    progress_bar = True
)
ppo_eval_args = dict(
    n_eval_episodes = 10
)
ppo_eval_callback_args = dict(
    best_model_save_path = os.path.join(ppo_log_root, "ckpt"),
    log_path = os.path.join(ppo_log_root, "eval_log"),
    eval_freq = 500,
    deterministic = True,
    render = False
)

default_policy_constructor = lambda _, __: "MlpPolicy"

def run_ppo(env: EnvInstanceT, eval_env: EnvInstanceT,
        policy_constructor: Callable[[AlgoT, EnvT], PolicyInstanceT] = default_policy_constructor,
        algo_args: Dict[str, Any] = ppo_algo_args,
        learn_args: Dict[str, Any] = ppo_learn_args,
        eval_args: Dict[str, Any] = ppo_eval_args,
        eval_callback_args: Dict[str, Any] = ppo_eval_callback_args):

    policy = policy_constructor(PPO, env.__class__)
    model = PPO(
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
