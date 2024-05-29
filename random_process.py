from typing import TypeVar, Iterable, Generic
from abc import abstractmethod

from function import Distribution
from utils import TimeSeriesMeta


StateT = TypeVar("StateT")


class RandomProcess(Generic[StateT], metaclass = TimeSeriesMeta):

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> StateT:
        raise NotImplementedError

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

    def state(self) -> StateT:
        return self._state

    def step(self) -> None:
        self._state = next(self._iter).sample()
