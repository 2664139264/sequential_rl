from typing import TypeVar, Iterable, Generic, Optional, Callable
from abc import abstractmethod

import numpy as np

import gymnasium as gym

from sequential_rl.envs.function import Distribution, DiscreteIntegerDistribution
from sequential_rl.envs.utils import TimeSeriesMeta, WithHistoryMeta, HistoryT, merge_meta


StateT = TypeVar("StateT", covariant = True)


class RandomProcessStem(Generic[StateT]):

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def state(self) -> Optional[StateT]:
        return self._state

    def observation(self) -> Optional[StateT]:
        return self.state()

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError


class RewardProcessStem(RandomProcessStem[StateT]):
    def reward(self) -> float:
        return self._reward


ActionT = TypeVar("ActionT", covariant = True)


class DecisionProcessStem(RewardProcessStem[StateT], Generic[StateT, ActionT]):
    def action(self) -> ActionT:
        return self._action


TimeSeriesWithHistoryMeta = merge_meta(TimeSeriesMeta, WithHistoryMeta)


class RandomProcess(RandomProcessStem[StateT], metaclass = TimeSeriesWithHistoryMeta):
    pass


class RewardProcess(RewardProcessStem[StateT], metaclass = TimeSeriesWithHistoryMeta):
    pass


class DecisionProcess(DecisionProcessStem[StateT, ActionT], metaclass = TimeSeriesWithHistoryMeta):
    pass


class IndependentProcess(RandomProcess[StateT]):
    
    def __init__(self, distributions: Iterable[Distribution[StateT]]):
        super().__init__()
        self._distributions = tuple(distributions)
        self._iter = iter(self._distributions)
        self._state = next(self._iter)

    def reset(self) -> None:
        self._iter = iter(self._distributions)
        self._state = next(self._iter)

    def step(self) -> None:
        self._state = next(self._iter).sample()


class MarkovProcess(RandomProcess[StateT]):
    
    def __init__(self,
            init_distribution: Distribution[StateT],
            transition_distribution: Callable[[StateT], Distribution[StateT]]):

        super().__init__()
        self._init_distribution = init_distribution
        self._transition_distribution = transition_distribution
        self._state = self._init_distribution.sample()

    def reset(self) -> None:
        self._state = self._init_distribution.sample()

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


class MarkovRewardProcess(MarkovProcess[StateT], RewardProcess[StateT]):

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

    def step(self) -> None:
        prev_state = self._state
        self._state = self._transition_distribution(prev_state).sample()
        self._reward = self._reward_distribution(prev_state, self._state).sample()


class MarkovDecisionProcess(MarkovRewardProcess[StateT], DecisionProcess[StateT, ActionT]):
    
    def __init__(self,
            init_distribution: Distribution[StateT],
            transition_distribution: Callable[[StateT, ActionT], Distribution[StateT]],
            reward_distribution: Callable[[StateT, ActionT, StateT], Distribution[float]]):        
        super().__init__(init_distribution, transition_distribution, reward_distribution)
        self._action = None

    def step(self, action: ActionT) -> None:
        prev_state = self._state
        self._state = self._transition_distribution(prev_state, action).sample()
        self._reward = self._reward_distribution(prev_state, action, self._state).sample()
        self._action = action


class GymEnvToDecisionProcessAdapter(DecisionProcess[StateT, ActionT]):

    def __init__(self, gym_env: gym.Env):
        self.env = gym_env
        self._state, self._info = self.env.reset()
        self._action, self._reward = None, None

    def reset(self) -> None:
        self._state, self._info = self.env.reset()
        self._action, self._reward = None, None

    def step(self, action: ActionT) -> None:
        self._action = action
        self._state, self._reward, terminated, truncated, info = self.env.step(action)
        self._info = {"terminated": terminated, "truncated": truncated, "info": info}

    def info(self) -> HistoryT:
        return self._info


class DecisionProcessWithPolicy(RewardProcess[StateT]):
    def __init__(self,
            decision_process: DecisionProcess[StateT, ActionT],
            policy: Callable[[StateT], Distribution[ActionT]]):
        self.decision_process = decision_process
        self.policy = policy
        self.decision_process.reset()

    def reset(self) -> None:
        self.decision_process.reset()
        
    def step(self) -> None:
        self.decision_process.step(self.policy(self.state()).sample())

    def state(self) -> StateT:
        return self.decision_process.state()
    
    def reward(self) -> float:
        return self.decision_process.reward()


class DecisionProcessAggregated(DecisionProcess[StateT, ActionT]):
    
    def __init__(self,
            decision_process: DecisionProcess[StateT, ActionT],
            aggregator: Callable[[HistoryT], StateT]):
        self.decision_process = decision_process
        self.aggregator = aggregator
        self.decision_process.reset()

    def reset(self) -> None:
        self.decision_process.reset()

    def step(self, action: ActionT) -> None:
        self.decision_process.step(action)

    def state(self) -> StateT:
        return self.aggregator(self.decision_process.history())

    def action(self) -> ActionT:
        return self.decision_process.action()

    def reward(self) -> float:
        return self.decision_process.reward()


# TODO：添加随机过程类到env类的适配接口，因为需要适配 stable_baselines 库。
# TODO：随机过程类包装env时，需要在done或者truncated时候抛出异常，或者通过info接口获取这些信息。

class DecisionProcessToGymEnvAdapter(gym.Env):
    pass