from abc import abstractmethod
from scipy import sparse
import numpy as np

class Grid:
  """
  Generic 1-dimensional abstract grid class.
  Features:
   - integration
   - differentiation
   - todo: interpolation 
  To be implemented by specific types of grids.
  """
  def __init__(self, boundary_condition: str, xmin: float, xmax: float, num: int) -> None:
    """
    bc - type of boundary condition: one of 'vanishing', 'periodic', 'open'
    xmin - smallest grid point 
    xmax - greatest grid point 
    num - number of grid points
    cmin - smallest point that belongs to the extend of the grid
    cmax - greatest point that belongs to the extend of the grid
    """
    if not boundary_condition in ['vanishing', 'periodic', 'open']: 
      raise NotImplementedError('unknown boundary condition `'+self.bc+'`!')

    self.bc = boundary_condition
    self.num = num
    self.xmin = xmin
    self.xmax = xmax

  @abstractmethod
  def integrate(self) -> None:
    pass

  @abstractmethod
  def first_deriv(self) -> None:
    pass

  @abstractmethod
  def second_deriv(self) -> None:
    pass


class UniformGrid(Grid):
  """
  Implementation with uniform grid spacing.
  """
  def __init__(self,  boundary_condition: str, xmin: float, xmax: float, num: int) -> None:
    super().__init__(boundary_condition, xmin,xmax, num)
    self.dx = (xmax-xmin)/num
    self.x  = lambda i: (i+0.5)*self.dx + self.xmin
    self.points = np.array([self.x(_i) for _i in range(self.num)])

  def integrate(self, func):
    return np.sum(func)*self.dx

  def first_deriv(self):
    """ 
    operator of the first finite-difference derivative
    finite differences like (-1,0,+1)/2/dx
    """
    result = sparse.lil_matrix((self.num, self.num))
    const = .5/self.dx
    result.setdiag(values=-const, k=-1)
    result.setdiag(values=+const, k=+1)
    if self.bc=='vanishing':
      result[ 0, :] = 0.
      result[-1, :] = 0.
    elif self.bc =='periodic':
      result[ 0, -1] = -const
      result[-1,  0] = +const
    elif self.bc =='open':
      result[0, :2] = -2*const, 2*const
      result[-1, -2:] = -2*const, 2*const
    else: 
      raise NotImplementedError('unknown boundary condition `'+self.bc+'`!')
    return result

  def second_deriv(self):
    """
    operator of the second finite-difference derivative 
    finite differences like (-1,2,-1)/dx**2
    """
    result = sparse.lil_matrix((self.num, self.num))
    const = 1./self.dx**2
    result.setdiag(values=const, k=-1)
    result.setdiag(values=-2.*const, k=0)
    result.setdiag(values=const, k=+1)
    if self.bc=='vanishing':
      result[ 0, :] = 0.
      result[-1, :] = 0.
    elif self.bc =='periodic':
      result[0, -1] = const
      result[-1,  0] = const
    elif self.bc == 'open':
      result[0, :3] = const, -2.*const, const
      result[-1, -3:] = const, -2.*const, const
    else: 
      raise NotImplementedError('unknown boundary condition `'+self.bc+'`!')
    return result
          

# class NUGrid(Grid):
#   def __init__(self, bc:str, xgrid:np.ndarray) -> None:
#     super().__init__(bc, np.min(xgrid), np.max(xgrid), xgrid.size)
#     self.xgrid = xgrid
#     self.dx = np.diff(xgrid)
#     self.x = lambda i: xgrid[i]
