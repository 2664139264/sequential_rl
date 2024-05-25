from abc import ABCMeta, abstractmethod
from typing import Callable, Hashable, TypeVar, Type


N = TypeVar("N", bound = int)
State = TypeVar("State", bound = Hashable)
StateDistribution = Callable[[State], float]


class RandomProcess(metaclass=ABCMeta):

    @abstractmethod
    def reset(self, distribution: StateDistribution) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> State:
        raise NotImplementedError
    
    @abstractmethod
    def terminate(self) -> bool:
        raise NotImplementedError


class FiniteStateMarkovProcess(RandomProcess):
    """
    Finite-state, first-order time-invariant Markov process.    
    """
    def __init__(self,
            n: int,
            transition_distribution: StateTransitionDistribution[int]):

        self.transition_matrix = 