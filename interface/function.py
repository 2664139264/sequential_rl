from abc import ABCMeta, abstractmethod
from typing import Callable, Generic, TypeVar, Optional

from gymnasium.spaces import Space


DomainVT = TypeVar("DomainVT", covariant = True)
CodomainVT = TypeVar("CodomainVT", covariant = True)


class Function(Generic[DomainVT, CodomainVT]):
    
    def __init__(self,
            func: Callable[[DomainVT], CodomainVT],
            domain: Space[DomainVT]):
        self._domain = domain
        self._func = func

    @property
    def support(self) -> Space[DomainVT]:
        return self._domain

    def __call__(self, x: DomainVT) -> CodomainVT:
        return self._func(x)


class ConstantFunction(Function[DomainVT, CodomainVT]):
    
    def __init__(self,
            value: CodomainVT,
            domain: Optional[Space[DomainVT]] = None):
        self._domain = domain
        self._value = value
        
    def __call__(self, _: DomainVT) -> CodomainVT:
        return self._value


class DiracDeltaFunction(Function[DomainVT, CodomainVT]):
    
    def __init__(self,
            x: DomainVT,
            value: CodomainVT,
            default_value: CodomainVT,
            domain: Optional[Space[DomainVT]] = None):
        self._domain = domain
        self._x = x
        self._value = value 
        self._default_value = default_value
        
    def __call__(self, x: DomainVT) -> CodomainVT:
        return self._value if self._x == x else self._default_value


class Distribution(Function[DomainVT, float], metaclass = ABCMeta):

    def prob(self, x: DomainVT) -> float:
        return self(x)
    
    @abstractmethod
    def sample(self) -> DomainVT:
        raise NotImplementedError