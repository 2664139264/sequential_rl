from itertools import repeat

import gymnasium as gym

import gymnasium.spaces as sp

from function import Function
from domain import *

from random_process import *

if __name__ == "__main__":
    
    rp = IndependentProcess((gym.spaces.Box(i,i) for i in range(1, 10)))
    
    
    for _ in range(100):
        print(rp.time())
        print(rp.state())
        
        rp.step()
        
    
    x = sp.Dict({"d":gym.spaces.Box(3,4,shape=(3,))})
    y = sp.Tuple((x, x, gym.spaces.Discrete(2)))
    
    print(x.sample(), y.sample())
    print(x.dtype, y.dtype)
    
    # print(dir(x), dir(y), sep="\n")
    
    

    
    f = Function(lambda x: x, sp.Dict({"f": sp.Box(0,1)}))

    print(f.support, f({"f":(0,)}))
    
    print(UniversalDomain() is UniversalDomain())

# class MyMeta(type):
#     def __new__(cls, name, bases, dct):
#         print(f"Creating class {name}")
#         dct['class_var'] = 'I am a class variable'
#         return super().__new__(cls, name, bases, dct)

#     def __init__(cls, name, bases, dct):
#         print(f"Initializing class {name}")
#         super().__init__(name, bases, dct)

#     def __call__(cls, *args, **kwargs):
#         print(f"Calling {cls.__name__} with args: {args} and kwargs: {kwargs}")
#         instance = super().__call__(*args, **kwargs)
#         return instance

# class MyClass(metaclass=MyMeta):
    
#     print("In MyClass")
#     def __init__(self, value):
#         self.value = value


# # 测试元类行为
# print("Defining class MyClass")
# print("----")
# mc = MyClass(10)
# print("----")
# print(f"Instance value: {mc.value}")
# print(f"Class variable: {MyClass.class_var}")


# class CaptureCallMeta(type):
#     def __new__(cls, name, bases, dct):
#         for key, value in dct.items():
#             if callable(value):
#                 dct[key] = cls.wrap_method(value)
#         return super().__new__(cls, name, bases, dct)
    
#     @staticmethod
#     def wrap_method(method):
#         def wrapper(*args, **kwargs):
#             print(f"Calling method {method.__name__} with args: {args} and kwargs: {kwargs}")
#             result = method(*args, **kwargs)
#             print(f"Method {method.__name__} returned {result}")
#             return result
#         return wrapper

# class MyClass(metaclass=CaptureCallMeta):
#     def method1(self, x):
#         return x * 2

#     def method2(self, y):
#         return y + 3

# obj = MyClass()
# obj.method1(10)
# obj.method2(20)