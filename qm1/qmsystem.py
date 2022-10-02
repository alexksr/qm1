import numpy as np
from qm1.grid import *

class QMSystem:
  def __init__(self, pot, grid:Grid, mass:float=1) -> None:
    # set potential 
    if callable(pot):
      self.pot = np.array([pot(_x) for _x in list(grid.points)])
    else:
      print('cannot call `pot`! init to zero potential')
      self.pot = np.zeros([grid.num])

    # define mass (which is 1 in atomic units for the electron)
    self.mass = mass

    # store the grid
    self.grid = grid

def ConstPot(const: float = 0.):
  """ Returns a vanishing (or constant) potential, solutions will be plain wave-like. """
  return lambda x: const

def StepPot(xstep: float = 0., ystep: float = 1.):
  """ Returns a step potential `ystep if x<=xstep else 0.` """
  return lambda x: ystep if x <= xstep else 0.

def BarrierPot(xstart: float = 0., xstop: float = 1., vstep: float = 1.):
  """ Returns a potential barrier `ystep if xstart<=x<=xstop else 0.` """
  return lambda x: vstep if xstart <= x <= xstop else 0.

def HarmonicPot(omega: float = 1., mass: float = 1):
  """ Returns a harmonic potential with parameter `omega`"""
  return lambda x: 0.5*mass*omega**2*x**2

def DeltaPot(grid: Grid, x0: float = 0., v0: float = 1.):
  """ 
  Returns a delta-like potential with strength `v0` at position `x0`.
  REMARK: Since the delta-distribution cannot be represented on a grid, use its box-like analogon.
  TODO: Only implemented for uniform grids, non-uniform grids need extra care for the length (1d-volume) of the grid cell.
  """
  return lambda x: -v0/grid.dx if -grid.dx <= x-x0 <= grid.dx else 0.

def DoubleDeltaPot(grid: Grid, x0: float = 0., v0: float = 1., x1: float = 1., v1: float = 1.):
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

def InterpolatePot(xs, vs):
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
