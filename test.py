# import gymnasium as gym

# import gymnasium.spaces as sp


# if __name__ == "__main__":
    
#     x = sp.Dict({"d":gym.spaces.Box(3,4,shape=(3,))})
#     y = sp.Tuple((x, x, gym.spaces.Discrete(2)))
    
#     print(x.sample(), y.sample())
#     print(x.dtype, y.dtype)
    
#     # print(dir(x), dir(y), sep="\n")
    
    
#     from function import Function
#     from domain import *
    
#     f = Function(lambda x: x, sp.Dict({"f": sp.Box(0,1)}))

#     print(f.support, f({"f":(0,)}))
    
#     print(UniversalDomain() is UniversalDomain())

class MyMeta(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name}")
        dct['class_var'] = 'I am a class variable'
        return super().__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct):
        print(f"Initializing class {name}")
        super().__init__(name, bases, dct)

    def __call__(cls, *args, **kwargs):
        print(f"Calling {cls.__name__} with args: {args} and kwargs: {kwargs}")
        instance = super().__call__(*args, **kwargs)
        return instance

class MyClass(metaclass=MyMeta):
    
    print("In MyClass")
    def __init__(self, value):
        self.value = value


# 测试元类行为
print("Defining class MyClass")
print("----")
mc = MyClass(10)
print("----")
print(f"Instance value: {mc.value}")
print(f"Class variable: {MyClass.class_var}")