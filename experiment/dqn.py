from typing import Callable, Dict, Any
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from utils import AlgoT, EnvT, PolicyInstanceT, EnvInstanceT


dqn_algo_args = dict(verbose=1)
dqn_learn_args = dict(total_timesteps=int(1e3), progress_bar=True)
dqn_eval_args = dict(n_eval_episodes=10)

default_policy_constructor = lambda _, __: "MlpPolicy"

def run_dqn(env: EnvInstanceT,
        policy_constructor: Callable[[AlgoT, EnvT], PolicyInstanceT] = default_policy_constructor,
        algo_args: Dict[str, Any] = dqn_algo_args,
        learn_args: Dict[str, Any] = dqn_learn_args,
        eval_args: Dict[str, Any] = dqn_eval_args):
    policy = policy_constructor(DQN, env.__class__)
    model = DQN(policy, env, **algo_args)
    model.learn(**learn_args)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), **eval_args)
    return mean_reward, std_reward
