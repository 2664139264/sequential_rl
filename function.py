from abc import ABCMeta, abstractmethod
from typing import Callable, Generic, TypeVar, Iterable

import numpy as np

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


class DiracDeltaDistribution(DiracDeltaFunction[DomainVT, float], Distribution[DomainVT]):

    def __init__(self, x: DomainVT):
        super().__init__(x, 1, 0)
        self._x = x
  
    def sample(self) -> DomainVT:
        return self._x


class DiscreteDistribution(Distribution[DomainVT]):

    def __init__(self, elements: Iterable[DomainVT], p: Iterable[float]):
        self._elements = tuple(elements)
        self._p = tuple(p)
        
    def sample(self) -> DomainVT:
        return np.random.choice(self._elements, p = self._p)


class DiscreteIntegerDistribution(DiscreteDistribution[int]):

    def __init__(self, p: Iterable[float]):
        super().__init__(range(len(p)), p)


def expectation(distribution: Distribution[float], sample_num = 1) -> float:
    return (
        distribution.expectation() if hasattr(distribution, expectation.__name__)
        else np.average(distribution.sample() for _ in range(sample_num))
    )

