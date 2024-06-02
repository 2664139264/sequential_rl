from typing import TypeVar, Iterable, Generic, Optional
from abc import abstractmethod

import numpy as np

from function import Distribution, DiscreteDistribution
from utils import TimeSeriesMeta, WithHistoryMeta, merge_meta


StateT = TypeVar("StateT")


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
        self._state = None

    def state(self) -> Optional[StateT]:
        return self._state

    def step(self) -> None:
        self._state = next(self._iter).sample()


class FiniteStateMarkovProcess(RandomProcess[int]):
    
    def __init__(self, init_distribution: DiscreteDistribution[int], transition_matrix: np.ndarray):
        super().__init__()
        self._init_distribution = init_distribution
        self._transition_matrix = transition_matrix
        assert (
            len(transition_matrix.shape) == 2
            and transition_matrix.shape[0] == transition_matrix.shape[1] == len(init_distribution)
        )
        self._state = None
        self._all_states = range(len(init_distribution))
        
    def reset(self) -> None:
        self._state = None
        
    def state(self) -> Optional[int]:
        return self._state

    def step(self) -> None:

        self._state = (
            self._init_distribution if self._state is None
            else DiscreteDistribution(self._all_states, self._transition_matrix[self._state])
        ).sample()
