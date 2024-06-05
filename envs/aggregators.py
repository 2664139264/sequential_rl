from typing import Iterable, Callable, TypeVar

from sequential_rl.envs.function import universal_identity_function as identity
from sequential_rl.envs.random_process import StateT, ActionT
from sequential_rl.envs.utils import HistoryT


def state_extractor(history: HistoryT) -> Iterable[StateT]:
    return map(lambda record: record["state"], history)

def action_extractor(history: HistoryT) -> Iterable[ActionT]:
    return map(lambda record: record["action"], history)

def reward_extractor(history: HistoryT) -> Iterable[float]:
    return map(lambda record: record["reward"], history)


def sum_aggregator(
        state_seq: Iterable[StateT],
        seq_sum_op: Callable[[Iterable[StateT]], StateT] = sum
    ) -> StateT:
    return seq_sum_op(state_seq)


StateWeightT = TypeVar("StateWeightT", covariant = True)


def weighted_sum_aggregator(
        state_seq: Iterable[StateT],
        weight_seq: Iterable[StateWeightT],
        seq_sum_op: Callable[[Iterable[StateT]], StateT] = sum,
        prod_op: Callable[[StateT, StateWeightT], StateT]= lambda a, b: a * b) -> StateT:
    
    return sum_aggregator(
        (prod_op(s, w) for s, w in zip(state_seq, weight_seq)),
        seq_sum_op
    )


def latest_aggregator(
        state_seq: Iterable[StateT],
        map_op: Callable[[StateT], StateT] = identity):
    return map_op(
        (state_seq if hasattr(state_seq, "__getitem__") else tuple(state_seq))[-1]
    )


