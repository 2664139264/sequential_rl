import gymnasium as gym

INIT_STATE_POS = 0
STATE_POS = 1


def extract_states(history):
    return (
        history[0][INIT_STATE_POS],
        *(h[STATE_POS] for h in history[1:])
    )


def aggr_latest(history):
    return history[-1]


def aggr_state_conv(history, state_aggregator = sum):
    result = list(aggr_latest(history))
    aggregated_state = state_aggregator(extract_states(history))
    result[STATE_POS if len(history) > 1 else INIT_STATE_POS] = aggregated_state
    return result


class AggregatedEnv(gym.Env):
    
    def __init__(self,
            id,
            aggregator = aggr_state_conv):
        self._env = gym.make(id)
        self._aggregator = aggregator
        
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, *args, **kwargs):
        reset_info = self._env.reset(*args, **kwargs)
        self._history = [reset_info]
        return self._aggregator(self._history)
    
    def step(self, action):
        step_return = self._env.step(action)
        self._history.append((action, *step_return))
        return self._aggregator(self._history)[1:]
    
    def render(self, *args, **kwargs):
        self._env.render(*args, **kwargs)
