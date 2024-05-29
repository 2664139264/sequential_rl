from abc import ABCMeta


class SingletonMeta(type, metaclass = ABCMeta):

    _instances = dict()

    def __call__(cls, *args, **kwargs):
        
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        
        return cls._instances[cls]


# 先执行，再修改时间
class TimeSeriesMeta(type, metaclass = ABCMeta):

    def __new__(cls, name, bases, dct):

        def time(self):
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
