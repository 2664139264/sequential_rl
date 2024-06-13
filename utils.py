from typing import Type, Union
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy


EnvT = Type[GymEnv]
EnvInstanceT = Union[GymEnv, str]

AlgoT = Type[BaseAlgorithm]
AlgoInstanceT = Union[BaseAlgorithm, str]

PolicyT = Type[BasePolicy]
PolicyInstanceT = Union[BasePolicy, str]