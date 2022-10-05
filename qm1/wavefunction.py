from scipy import integrate
import numpy as np
from qm1.grid import *

class Wavefunction:
  """ 
  Generic class for a wave function on a 1-dimenional grid.
   - representation of the function via the values at the grid points.
   - arithmetic operations supported
   - statistical operations supported
  """
  def __init__(self, grid: Grid):
    self.grid  = grid
    self.func = np.zeros([self.grid.num])

  def __mul__(self, other) -> 'Wavefunction':
    """ multiply a Wavefunction by a constant """
    result = Wavefunction(self.grid)
    result.func = other * self.func
    return result

  def __add__(self, other: 'Wavefunction') -> 'Wavefunction':
    """ add two Wavefunctions """
    result = Wavefunction(self.grid)
    result.func = other.func + self.func
    return result
  def __sub__(self, other: 'Wavefunction') -> 'Wavefunction':
    """ add two Wavefunctions """
    result = Wavefunction(self.grid)
    result.func = self.func - other.func
    return result

  def norm(self):
    """ norm of the wavefunction """
    return np.sqrt(self.scalar_prod(self))
    
  def normalized(self):
    """ normalize the wavefunction to one """
    result = self
    result.func = self.func / self.norm()
    return result

  def conjugated(self):
    """ conjugate the wavefunction (complex conjugation)  """
    result = self
    result.func = np.conjugate(self.func)
    return result

  def scalar_prod(self, other:"Wavefunction")->complex:
    """ calc `<self|other>` """
    return self.grid.integrate(np.conjugate(self.func)*other.func)

  def set_via_array(self, func: np.ndarray):
    """ set the wavefunction to the given array of values `func` """
    self.func = func
    self = self.normalized()
    return None

  def set_via_func(self, func):
    """ 
    Set the wavefunction by evaluating the given function on the grid.
    Automatically do a normalization.
    """
    if callable(func):
      self.func = func(self.grid.points)
    else:
      raise TypeError('func not callable')
      self.func = 0.
    self = self.normalized()
    return None

  def expectation_value(self, operator:'Operator') -> complex:
    """ evaluates the expectation value of the operator with the wave function  `<self|op(self)>`"""
    return self.scalar_prod(operator(self))

  def variance(self, operator: 'Operator') -> complex:
    """ evaluates the variance of the operator with the wave function """
    expval = self.expectation_value(operator)
    dummy = operator(self) - self*expval
    return dummy.scalar_prod(dummy)

  def impose_boundary_condition(self):
    if self.grid.bc=='vanishing':
      self.func[0], self.func[-1] = 0., 0.
    elif self.grid.bc=='periodic' or self.grid.bc=='open':
      pass
    else:
      raise NotImplementedError('unknown boundary condition `'+self.bc+'`. Stop!')
    
  def get_observables(self, ops:list):
    """
    Return the expectation value and variance for each operator in the list of operators `ops`.
    """
    obs = []
    for _op in ops:
      obs.append( [self.expectation_value(_op), self.variance(_op)] )
    return np.array(obs)
  
  def evolve(self, tgrid:np.ndarray, op_rhs:'OperatorTD'): 
    """ propagate the wavefunction in time 
    - return a list `psis` of wave functions, where `psis[i] = psi(t_i)` with `t_i=i*dt`
    """
    # callable for the ivp solver
    def ipvfunc(t, vec): return op_rhs.sparse_mat(t) * vec

    # solve 
    data = integrate.solve_ivp(fun=ipvfunc, t_span=[tgrid[0], tgrid[-1]], y0=self.func, t_eval=tgrid, method="RK45")
  

    # make the raw output data to class wavefuctions again
    psis = []
    for _i in range(tgrid.size):
      psi=Wavefunction(self.grid)
      psi.set_via_array(data.y[:,_i])
      psis.append(psi)
    return psis, tgrid
   
  def show(self, file):
    """
    Save a graphical representation of the wave function to file.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(self.grid.points, np.abs(self.func), label='density')
    ax.plot(self.grid.points, np.real(self.func), label='real part')
    ax.plot(self.grid.points, np.imag(self.func), label='imag part')
    fig.legend(loc='upper center', ncol=3)  # , bbox_to_anchor=(1., 1.))
    plt.savefig(file)
    plt.close()


def GaussianWavePackage(grid: Grid, mu: float = 0, sigma: float = 1, k: float=1):
  def prep_wf(x): return np.exp(-(x-mu)**2 / (2.0 * sigma**2)) * np.exp(-1j * k * x)
  wf = Wavefunction(grid)
  wf.set_via_func(prep_wf)
  return wf

