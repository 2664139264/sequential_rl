from abc import ABCMeta, abstractmethod
from typing import Callable, Generic, TypeVar

from gymnasium.spaces import Space

from domain import UniversalDomain


DomainVT = TypeVar("DomainVT", covariant = True)
CodomainVT = TypeVar("CodomainVT", covariant = True)


class Function(Generic[DomainVT, CodomainVT]):
    
    def __init__(self,
            func: Callable[[DomainVT], CodomainVT],
            domain: Space[DomainVT] = UniversalDomain()):
        self._domain = domain
        self._func = func

    def support(self) -> Space[DomainVT]:
        return self._domain

    def __call__(self, x: DomainVT) -> CodomainVT:
        if not self._domain.contains(x):
            raise ValueError(f"Value {x} not in domain {self._domain}.")
        return self._func(x)


class ConstantFunction(Function[DomainVT, CodomainVT]):
    
    def __init__(self,
            value: CodomainVT,
            domain: Space[DomainVT] = UniversalDomain()):
        super().__init__(
            lambda _: value,
            domain
        )


class DiracDeltaFunction(Function[DomainVT, CodomainVT]):
    
    def __init__(self,
            x: DomainVT,
            value: CodomainVT,
            default_value: CodomainVT,
            domain: Space[DomainVT] = UniversalDomain()):
        super().__init__(
            lambda _x: value if _x == x else default_value,
            domain
        )


class IdentityFunction(Function[DomainVT, DomainVT]):
    
    def __init__(self,
            domain: Space[DomainVT] = UniversalDomain()):
        super().__init__(
            lambda x: x,
            domain
        )


class Distribution(Function[DomainVT, float], metaclass = ABCMeta):

    @abstractmethod
    def sample(self) -> DomainVT:
        raise NotImplementedError
