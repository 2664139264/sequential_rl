from typing import TypeVar, Iterable, Generic, Optional, Callable
from abc import abstractmethod

import numpy as np

import gymnasium as gym

from function import Distribution, DiscreteIntegerDistribution
from utils import TimeSeriesMeta, WithHistoryMeta, merge_meta


StateT = TypeVar("StateT", covariant=True)


class RandomProcess(Generic[StateT], metaclass = merge_meta(TimeSeriesMeta, WithHistoryMeta)):

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> Optional[StateT]:
        raise NotImplementedError

    def observation(self) -> Optional[StateT]:
        return self.state()

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError


class IndependentProcess(RandomProcess[StateT]):
    
    def __init__(self, distributions: Iterable[Distribution[StateT]]):
        super().__init__()
        self._distributions = tuple(distributions)
        self._iter = iter(self._distributions)
        self._state = None

    def reset(self) -> None:
        self._iter = iter(self._distributions)
        self._state = next(self._iter)

    def state(self) -> Optional[StateT]:
        return self._state

    def step(self) -> None:
        self._state = next(self._iter).sample()


class MarkovProcess(RandomProcess[StateT]):
    
    def __init__(self,
            init_distribution: Distribution[StateT],
            transition_distribution: Callable[[StateT], Distribution[StateT]]):
        
        super().__init__()
        self._init_distribution = init_distribution
        self._transition_distribution = transition_distribution
        self._state = None

    def reset(self) -> None:
        self._state = self._init_distribution.sample()
        
    def state(self) -> Optional[StateT]:
        return self._state
    
    def step(self) -> None:
        self._state = self._transition_distribution(self._state).sample()


class FiniteStateMarkovProcess(MarkovProcess[int]):
    
    def __init__(self,
            init_distribution: DiscreteIntegerDistribution,
            transition_matrix: np.ndarray):
        assert(
            len(transition_matrix.shape) == 2
            and len(init_distribution) == transition_matrix.shape[0] == transition_matrix.shape[1]
        )
        init_dist = DiscreteIntegerDistribution(init_distribution)
        next_dist = [DiscreteIntegerDistribution(dist) for dist in transition_matrix]
        super().__init__(
            init_dist,
            lambda state: next_dist[state]
        )


class MarkovRewardProcess(MarkovProcess[StateT]):
    
    def __init__(self,
            init_distribution: Distribution[StateT],
            transition_distribution: Callable[[StateT], Distribution[StateT]],
            reward_distribution: Callable[[StateT, StateT], Distribution[float]]):
        super().__init__(init_distribution, transition_distribution)
        self._reward_distribution = reward_distribution
        self._reward = None
        
    def reset(self) -> None:
        super().reset()
        self._reward = None

    def reward(self) -> Optional[float]:
        return self._reward
    
    def step(self) -> None:
        prev_state = self._state
        self._state = self._transition_distribution(prev_state).sample()
        self._reward = self._reward_distribution(prev_state, self._state).sample()


ActionT = TypeVar("ActionT", covariant = True)


class DecisionProcess(RandomProcess[StateT], Generic[StateT, ActionT]):
    pass

class MarkovDecisionProcess(MarkovRewardProcess[StateT], DecisionProcess[StateT, ActionT]):
    
    def __init__(self,
            init_distribution: Distribution[StateT],
            transition_distribution: Callable[[StateT, ActionT], Distribution[StateT]],
            reward_distribution: Callable[[StateT, ActionT, StateT], Distribution[float]]):        
        super().__init__(init_distribution, transition_distribution, reward_distribution)
        self._action = None

    def reset(self) -> None:
        super().reset()
        self._action = None
        
    def action(self) -> ActionT:
        return self._action
    
    def step(self, action: ActionT) -> None:
        prev_state = self._state
        self._state = self._transition_distribution(prev_state, action).sample()
        self._reward = self._reward_distribution(prev_state, action, self._state).sample()
        self._action = action


# This should be Gym env to decision process adapter
class GymEnvToMarkovDecisionProcessAdapter(MarkovDecisionProcess[StateT, ActionT]):

    info_keys = ("terminated", "truncated", "info")

    def __init__(self, gym_env: gym.Env):
        self.env = gym_env
        self._state, self._action, self._reward, self._info = None, None, None, None
        
    def reset(self) -> None:
        self._state, self._info = self.env.reset()
        self._action, self._reward = None, None

    def step(self, action: ActionT) -> None:
        self._action = action
        self._state, self._reward, terminated, truncated, info = self.env.step(action)
        self._info = {"terminated": terminated, "truncated": truncated, "info": info}


# this should be decision process with policy
class MarkovDecisionProcessWithPolicy(MarkovRewardProcess[StateT]):
    def __init__(self, markov_decision_process: MarkovDecisionProcess[StateT, ActionT], policy: Callable[[StateT], Distribution[ActionT]]):
        self.mdp = markov_decision_process
        self.policy = policy
        self._state, self._reward = None, None
        
    def reset(self) -> None:
        self.mdp.reset()
        self._state, self._reward = self.mdp.state(), self.mdp.reward()
        
    def step(self) -> None:
        self.mdp.step(self.policy(self._state).sample())
        self._state, self._reward = self.mdp.state(), self.mdp.reward()


# MarkovRewardProcess should have base reward process
class DecisionProcessWithAggregator:
    pass