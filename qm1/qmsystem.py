import numpy as np
from qm1.grid import *
from typing import Callable

class QMSystem:
  """
  Defines the quantum mechanical system:
    - grid
    - mass of the particle
    - stationary potential
    - optionally: time-dependent potential
  """

  def __init__(self,  grid: Grid, stat_pot: Callable[[float], float], td_pot: Callable[[float, float], float] = None, mass: float = 1) -> None:
    # store the grid
    self.grid = grid
    
    # set potential 
    if callable(stat_pot):
      self.stat_pot = stat_pot
    else:
      print('cannot call `stat_pot`! init to zero potential')
      self.stat_pot = lambda _x: 0.

    # define mass (which is 1 in atomic units for the electron)
    self.mass = mass

    # td potential
    if td_pot and callable(td_pot):
      self.td_pot = td_pot
    
    # full potential
    def _full_pot(_x,_t):
      return self.td_pot(_x, _t)+self.stat_pot(_x)
    self.full_pot = _full_pot


def ConstPot(const: float = 0.) -> Callable[[float], float]:
  """ Returns a vanishing (or constant) potential, solutions will be plain wave- or box-like. """
  return lambda x: const


def StepPot(xstep: float = 0., ystep: float = 1.) -> Callable[[float], float]:
  """ Returns a step potential `ystep if x<=xstep else 0.` """
  return lambda x: ystep if x <= xstep else 0.

def BarrierPot(xstart: float = 0., xstop: float = 1., vstep: float = 1.)->Callable[[float],float]:
  """ Returns a potential barrier `ystep if xstart<=x<=xstop else 0.` """
  return lambda x: vstep if xstart <= x <= xstop else 0.


def HarmonicPot(omega: float = 1., mass: float = 1) -> Callable[[float], float]:
  """ Returns a harmonic potential with parameter `omega`"""
  return lambda x: 0.5*mass*omega**2*x**2


def DeltaPot(grid: Grid, x0: float = 0., v0: float = 1.) -> Callable[[float], float]:
  """ 
  Returns a delta-like potential with strength `v0` at position `x0`.
  REMARK: Since the delta-distribution cannot be represented on a grid, use its box-like analogon.
  TODO: Only implemented for uniform grids, non-uniform grids need extra care for the length (1d-volume) of the grid cell.
  """
  return lambda x: -v0/grid.dx if -grid.dx <= x-x0 <= grid.dx else 0.


def DoubleDeltaPot(grid: Grid, x0: float = 0., v0: float = 1., x1: float = 1., v1: float = 1.) -> Callable[[float], float]:
  """ 
  Returns a double delta-like potential with strength `v0, v1` at position `x0, x1`.
  REMARK: Since the delta-distribution cannot be represented on a grid, use its box-like analogon.
  TODO: Only implemented for uniform grids, non-uniform grids need extra care for the length (1d-volume) of the grid cell.
  """
  def pot(x):
    _pot = 0.
    if -grid.dx <= x-x0 <= grid.dx:
      _pot += -v0/grid.dx
    if -grid.dx <= x-x1 <= grid.dx:
      _pot += -v1/grid.dx
    return _pot
  return pot


def InterpolatePot(xs, vs) -> Callable[[float], float]:
  """
  Returns a potential with linear interpolation between the given points (e.g. step potential, triangle, saw tooth,...), otherwise zero.
  REMARK: the list of xs must be ordered.
  """
  def pot(x):
    # get points to the left and right of x
    ileft = min(sorted(range(len(xs)), key=lambda _i: abs(xs[_i]-x))[:2])
    iright = max(sorted(range(len(xs)), key=lambda _i: abs(xs[_i]-x))[:2])
    xleft, vleft = xs[ileft], vs[ileft]
    xright, vright = xs[iright], vs[iright]
    # set to zero or interpolate
    if x < min(xs) or x > max(xs):
      _pot = 0
    else:
      _pot = (x-xleft)/(xright-xleft)*(vright-vleft) + vleft
    return _pot
  return pot


def ZeroTDPot(omega: float = 2*np.pi, amplitude: float = 1., phase: float = 0.) -> Callable[[float,float], float]:
  """
  Return the callable for a vanishing potential.
  """
  return lambda x,t:  0.


def DipolTDPot(omega: float = 2*np.pi, k: float = 2*np.pi, amplitude: float = 1., phase: float = 0.) -> Callable[[float, float], float]:
  """
  Return the callable for a periodic time dependend dipol potential.
  $$
  V(x,t) = A \sin(\omega t - kx +\varphi)
  $$
  """
  return lambda x, t:  np.sin(omega*t-k*x+phase) * amplitude


def GrowingBarrierTDPot(tstart: float = 0., tstop: float = 1., vstep: float = 1., xstart: float = -1., xstop: float = +1.) -> Callable[[float, float], float]:
  """
  Return the callable for a growing time dependend barrier potential.
  In the interval `xmin < x < xmax` a barrier (or pod) is growing from depth 0 to `max_depth` in time interval `tmin < t < tmax`
  """
  def _func(x,t):
    barrier = vstep if xstart < x < xstop else 0.
    ftime = 0. if t < tstart else ( (t-tstart)/(tstop-tstart) if tstart < t < tstop else 1. )
    return barrier * ftime
  return _func
