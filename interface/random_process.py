from abc import ABCMeta, abstractmethod
from typing import Hashable, TypeVar


State = TypeVar("State", bound = Hashable)


class RandomProcess(metaclass=ABCMeta):

    @abstractmethod
    def reset(self, distribution) -> None:
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
