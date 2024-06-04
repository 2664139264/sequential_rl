from abc import ABCMeta

from typing import List, Type


class SingletonMeta(type, metaclass = ABCMeta):

    _instances = dict()

    def __call__(cls, *args, **kwargs):
        
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        
        return cls._instances[cls]


# 先执行，再修改时间
class TimeSeriesMeta(type, metaclass = ABCMeta):

    def __new__(cls, name, bases, dct):

        def time(self) -> int:
            return self._time

        def reset_wrapper(reset):
            def time_series_reset(self, *args, **kwargs):
                result = reset(self, *args, **kwargs)
                self._time = 0
                return result
            return time_series_reset

        def step_wrapper(step):
            def time_series_step(self, *args, **kwargs):
                result = step(self, *args, **kwargs)
                self._time += 1
                return result
            return time_series_step

        for method_name, method in dct.items():
            if method_name in {"__init__", "reset"}:
                dct[method_name] = reset_wrapper(method)
            elif method_name == "step":
                dct[method_name] = step_wrapper(method)

        dct["time"] = time

        return super().__new__(cls, name, bases, dct)


# 先执行，再修改历史
class WithHistoryMeta(type, metaclass = ABCMeta):

    history_keys = ("state", "action", "reward", "info")

    def __new__(cls, name, bases, dct):
        
        def history(self) -> List:
            return self._history
        
        def step_wrapper(step):
            def record_history_step(self, *args, **kwargs):
                result = step(self, *args, **kwargs)
                
                step_info = dict()
                for key in WithHistoryMeta.history_keys:
                    if hasattr(self, key):
                        step_info[key] = getattr(self, key)()

                self._history.append(step_info)
                return result
            return record_history_step

        def reset_wrapper(reset):
            def record_history_reset(self, *args, **kwargs):
                result = reset(self, *args, **kwargs)
                self._history = list()
                return result
            return record_history_reset

        for method_name, method in dct.items():
            if method_name in {"__init__", "reset"}:
                dct[method_name] = reset_wrapper(method)
            elif method_name == "step":
                dct[method_name] = step_wrapper(method)
        
        dct["history"] = history
        
        return super().__new__(cls, name, bases, dct)


def merge_meta(*meta_cls: Type) -> Type:
    merged_name = "".join(cls.__name__.removesuffix("Meta") for cls in meta_cls) + "Meta"
    return type(merged_name, meta_cls, {})
