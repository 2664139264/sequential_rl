from typing import TypeVar, Optional, Iterable, Generic

from function import Distribution

StateT = TypeVar("StateT")


# TODO: 时间记录功能可以与类实现解耦
class RandomProcess(Generic[StateT]):

    def __init__(self, distribution: Distribution[StateT]):
        self._time = 0
        self._init_distribution = distribution
        self._state = distribution.sample()

    def reset(self, distribution: Optional[Distribution[StateT]] = None) -> None:
        self._time = 0
        self._state = (self._init_distribution if distribution is None else distribution).sample()

    def state(self) -> StateT:
        return self._state

    def step(self) -> None:
        self._time += 1
        
    def time(self) -> int:
        return self._time
        

class IndependentProcess(RandomProcess[StateT]):
    
    def __init__(self, distributions: Iterable[Distribution[StateT]]):
        self._distributions = distributions
        self._distribution_iter = iter(self._distributions)
        super().__init__(next(self._distribution_iter))
        
    def reset(self) -> None:
        self._distribution_iter = iter(self._distributions)
        super().reset(self._init_distribution)
        
    def step(self) -> None:
        super().step()
        self._state = next(self._distribution_iter).sample()
