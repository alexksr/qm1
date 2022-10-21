from time import time
from copy import deepcopy
import numpy as np

_timit_num_runs = 10


def timeit_multiple(some_function):
  """ Timer for any function with printing and statistics.
  Repeats the timing and returns simple statistics on the delay.
  Uses safe handling of arg-altering functions.
  Returns only the last return value of `some_function`.
  """
  def wrapper(*args, **kwargs):
    dts = np.zeros((_timit_num_runs))
    for _i in range(_timit_num_runs):
      args_copied = deepcopy(args)
      kwargs_copied = deepcopy(kwargs)
      t1 = time()
      result = some_function(*args_copied, **kwargs_copied)
      dt = time()-t1
      dts[_i] = dt
    _mean = np.mean(dts)
    _min = np.min(dts)
    _max = np.max(dts)
    _sde = np.sqrt(np.var(dts))
    print('@timeit_multiple of func:  ', some_function.__name__)
    print('   after ', _timit_num_runs, ' runs:')
    print('   mean, min, max, var', _mean, _min, _max, _sde)
    return result
  return wrapper


def timeit_single(some_function):
  """ Timer for a single call of a function.
  returns `result_of_some_function, delay`
  """
  def wrapper(*args, **kwargs):
    ts = time()
    result = some_function(*args, **kwargs)
    dt = time()-ts
    return result, dt
  return wrapper
