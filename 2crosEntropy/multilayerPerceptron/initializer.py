import numpy as np

def get_fans(shape):
    # if is linear layer, shape is (input_dim, output_dim)
    # if is conv layer, shape is (out_channels, in_channels, kernel_size...)
    # conv = (64, 3, 3, 3) -> fan_in = 3*3*3 = 27, fan_out = 64*3*3 = 576
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

# define a base class, with __call__ method can let the instance be called like a function
class Initializer:
    def __call__(self, shape):
        return self.init(shape).astype(np.float32)

    # the child class should implement this method to return a numpy array of the given shape
    def init(self, shape):
        raise NotImplementedError()


class Norm(Initializer):
    def __init__(self, mean=0.0, std=1):
        self._mean = mean
        self._std = std

    def init(self, shape):
        # loc is mean, scale is std error, size is the output shape
        return np.random.normal(loc=self._mean, scale=self._std, size=shape)

class TruncatedNorm(Initializer):
    def __init__(self, low, high, mean=0.0, std=1):
        self._low = low
        self._high = high
        self._mean = mean
        self._std = std

    def init(self, shape):
        data = np.random.normal(loc=self._mean, scale=self._std, size=shape)
        while True:
            mask = (data > self._low) & (data < self._high) # True or False
            if mask.all():
                break
            # ~mask false
            data[~mask] = np.random.normal(loc=self._mean, scale=self._std, size=(~mask).sum())
        return data


class Uniform(Initializer):
    def __init__(self, a = 0, b = 1):
        self._a = a
        self._b = b

    def init(self, shape):
        return np.random.uniform(low=self._a, high=self._b, size=shape)

class Constant(Initializer):
    def __init__(self, value):
        self._value = value

    def init(self, shape):
        return np.full(shape=shape, fill_value=self._value)

class Zeros(Constant):
    def __init__(self):
        super().__init__(0.0)

class Ones(Constant):
    def __init__(self):
        super().__init__(1.0)


class XavierUniform(Initializer):
    def __init__(self, gain=1.0):
        self._gain = gain
    def init(self, shape):
        fan_in, fan_out = get_fans(shape)
        limit = self._gain * np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(low=-limit, high=limit, size=shape)

class XavierNorm(Initializer):
    def __init__(self, gain=1.0):
        self._gain = gain
    def init(self, shape):
        fan_in, fan_out = get_fans(shape)
        std = self._gain * np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(loc=0.0, scale=std, size=shape)


class HeUniform(Initializer):
    def __init__(self, gain=1.0):
        self._gain = gain
    def init(self, shape):
        fan_in, _ = get_fans(shape)
        limit = self._gain * np.sqrt(6 / fan_in)
        return np.random.uniform(low=-limit, high=limit, size=shape)

class HeNorm(Initializer):
    def __init__(self, gain=1.0):
        self._gain = gain
    def init(self, shape):
        fan_in, _ = get_fans(shape)
        std = self._gain * np.sqrt(2 / fan_in)
        return np.random.normal(loc=0.0, scale=std, size=shape)

