import gymnasium as gym

from sequential_rl.envs.random_process import GymEnvToDecisionProcessAdapter

MakeEnv = lambda name, version: GymEnvToDecisionProcessAdapter(gym.make(f"{name}-v{version}"))

CartPoleEnv = lambda version: MakeEnv("CartPole", version)

