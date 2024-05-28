from typing import Any

from gymnasium.spaces import Space

from utils import SingletonMeta


class UniversalDomain(Space, metaclass = SingletonMeta):

    def contains(self, _: Any) -> bool:
        return True
