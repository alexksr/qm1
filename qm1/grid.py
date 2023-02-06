from abc import abstractmethod
from scipy import sparse
import numpy as np


class Grid:
  """
  Generic 1-dimensional abstract grid class.
  Features integration and differentiation.

  Notes
  -----
  Abstract type: To be implemented by specific types of grids.

  Features
  --------
  - integration
  - differentiation
  - todo: interpolation 
  """
  def __init__(self, boundary_condition: str, xmin: float, xmax: float, num: int) -> None:
    """
    Initializes the grid class.

    Parameters
    ----------
    ``boundary_condition: str``
        type of boundary condition: one of 'vanishing', 'periodic', 'open'
    ``xmin: float``
        smallest grid point
    ``xmax: float``
        greatest grid point
    ``num: int``
        number of grid points    
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
  """ Implementation of a `Grid` class with uniform grid spacing. """
  def __init__(self,  boundary_condition: str, xmin: float, xmax: float, num: int) -> None:
    """ 
    Set up an uniform grid. 
    
    Parameters
    ----------
    ``boundary_condition: str``
      Boundary condition to impose. Choose from 'vanishing' (wave functions are identically zero at the boundary), 'periodic' (functions (and derivatives) are periodic at the boundary), 'open' (no conditions imposed).
    ``xmin: float``
      Smallest coordinate.
    ``xmax: float``
      Greatest coordinate.
    ``num: int``
      Number of grid points. 
    """
    super().__init__(boundary_condition, xmin,xmax, num)
    self.dx = (xmax-xmin)/num
    self.x  = lambda i: (i+0.5)*self.dx + self.xmin
    self.points = np.array([self.x(_i) for _i in range(self.num)])

  def integrate(self, func) -> float:
    """
    Integrate ``func`` on the grid by the simplest sum rule.

    Parameters
    ----------
    ``func : np.ndarray``
      Function/vector to integrate.

    Returns
    -------
    Returns the integral of function ``func`` over the interval of the grid in linear accuracy.
    """
    return np.sum(func)*self.dx

  def first_deriv(self) -> sparse.lil_matrix :
    """ 
    Return the operator of the first derivative in linear order using finite differences $(-1,0,+1)/2/dx$.
    
    Returns
    -------
    ``deriv : sparse.lil_matrix  ``
      Returns the sparse matrix representation of the first derivative.

    Notes
    -----
    The derivative matrix respects the chosen kind of boundary conditions.
    """
    deriv = sparse.lil_matrix((self.num, self.num))
    const = .5/self.dx
    deriv.setdiag(values=-const, k=-1)
    deriv.setdiag(values=+const, k=+1)
    if self.bc=='vanishing':
      deriv[ 0, :] = 0.
      deriv[-1, :] = 0.
    elif self.bc =='periodic':
      deriv[ 0, -1] = -const
      deriv[-1,  0] = +const
    elif self.bc =='open':
      deriv[0, :2] = -2*const, 2*const
      deriv[-1, -2:] = -2*const, 2*const
    else: 
      raise NotImplementedError('unknown boundary condition `'+self.bc+'`!')
    return deriv

  def second_deriv(self):
    """ 
    Return the operator of the second derivative in linear order using finite differences $(-1,2,-1)/dx**2$.
    
    Returns
    -------
    ``deriv2 : sparse.lil_matrix  ``
      Returns the sparse matrix representation of the second derivative.

    Notes
    -----
    The derivative matrix respects the chosen kind of boundary conditions.
    """
    deriv2 = sparse.lil_matrix((self.num, self.num))
    const = 1./self.dx**2
    deriv2.setdiag(values=const, k=-1)
    deriv2.setdiag(values=-2.*const, k=0)
    deriv2.setdiag(values=const, k=+1)
    if self.bc=='vanishing':
      deriv2[ 0, :] = 0.
      deriv2[-1, :] = 0.
    elif self.bc =='periodic':
      deriv2[0, -1] = const
      deriv2[-1,  0] = const
    elif self.bc == 'open':
      deriv2[0, :3] = const, -2.*const, const
      deriv2[-1, -3:] = const, -2.*const, const
    else: 
      raise NotImplementedError('unknown boundary condition `'+self.bc+'`!')
    return deriv2
          
# TODO
# class NUGrid(Grid):
#   def __init__(self, bc:str, xgrid:np.ndarray) -> None:
#     super().__init__(bc, np.min(xgrid), np.max(xgrid), xgrid.size)
#     self.xgrid = xgrid
#     self.dx = np.diff(xgrid)
#     self.x = lambda i: xgrid[i]
