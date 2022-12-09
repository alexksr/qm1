from matplotlib import colors as mcolors
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import Callable, Iterable, Union
from scipy import integrate
import numpy as np
from qm1.grid import *


_wf_anim_cm_phase_ticks = [-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi]
_wf_anim_cm_phase_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$']
_wf_anim_cm_norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
_wf_anim_cmap = cm.hsv
_wf_anim_cm_mappable = cm.ScalarMappable(norm=_wf_anim_cm_norm, cmap=_wf_anim_cmap)
_wf_figsize = (6.4, 4.8)

def mpl_absphase_plot(fig, ax, grid: Grid, func: np.ndarray):
  colors = _wf_anim_cmap(np.angle(func)/2./np.pi + 0.5)
  bar = ax.bar(grid.points, np.abs(func), color=colors, width=grid.dx)
  line,  = ax.plot(grid.points, np.abs(func), c='black', lw=3)
  return  bar, line

def mpl_absphase_colorbar(fig, ax, cax):
  cbar = fig.colorbar(mappable=_wf_anim_cm_mappable, ax=ax, cax=cax, ticks=_wf_anim_cm_phase_ticks, fraction=0.046, pad=0.04)
  cbar.set_ticklabels(_wf_anim_cm_phase_labels)
  return cbar

def mpl_absphase_plot_update(fig, ax, grid: Grid, func: np.ndarray, bar, line: Line2D):
  colors = _wf_anim_cmap(np.angle(func)/2./np.pi + 0.5)
  line.set_ydata(np.abs(func))
  fig.canvas.draw()
  for _i in range(grid.num):
    rect = bar[_i]
    _f = func[_i]
    _color = colors[_i]
    rect.set_height(np.abs(_f))
    rect.set(color=_color)
  fig.canvas.draw()

class Wavefunction:
  """ 
  Generic class for a wave function on a 1-dimenional grid.
   - representation of the function via the values at the grid points.
   - arithmetic operations supported
   - statistical operations supported
  """
  def __init__(self, grid: Grid):
    """
    Initializes the wave function class from a given grid.
    Parameters
    ----------
    grid: Grid
      Determines grid positions.
    """
    self.grid  = grid
    self.func = np.zeros([self.grid.num])

  def __mul__(self, other:Union[float, int, complex]) -> 'Wavefunction':
    """ Return the product of a wave function with a scalar. """
    result = Wavefunction(self.grid)
    result.func = other * self.func
    return result
  def __rmul__(self, other: Union[float, int, complex]) -> 'Wavefunction':
    """ Return the product of a wave function with a scalar. """
    result = Wavefunction(self.grid)
    result.func = self.func * other
    return result
  def __add__(self, other: 'Wavefunction') -> 'Wavefunction':
    """ Return the sum of two wave functions. """
    result = Wavefunction(self.grid)
    result.func = other.func + self.func
    return result
  def __sub__(self, other: 'Wavefunction') -> 'Wavefunction':
    """ Return the difference of two wave functions. """
    result = Wavefunction(self.grid)
    result.func = self.func - other.func
    return result

  def norm(self):
    """ Return the norm of the wave function. """
    return np.sqrt(self.scalar_prod(self))
    
  def normalized(self):
    """ Return a normalized wave function. """
    result = self
    result.func = self.func / self.norm()
    return result

  def conjugated(self):
    """ Return a complex conjugated wavefunction. """
    result = self
    result.func = np.conjugate(self.func)
    return result

  def scalar_prod(self, other:"Wavefunction")->complex:
    """ Return the scalar product of the two wave functions `<self|other>`. """
    return self.grid.integrate(np.conjugate(self.func)*other.func)

  def from_array(self, func: np.ndarray, normalize:bool=True):
    """ Set the  wave function values to the given vector values of `func` (and normalize by default). """
    self.func = func
    if self.grid.bc == 'vanishing':
      self.func[0] = 0.
      self.func[-1] = 0.
    if normalize: 
      self = self.normalized()
    return None

  def from_func(self, func:Callable[[float], Union[float,complex]] , normalize: bool = True):
    """Set the  wave function values by evaluating the given function on the grid (and normalize by default). """
    if callable(func):
      self.func = func(self.grid.points)
    else:
      raise TypeError('func not callable')
    if self.grid.bc == 'vanishing':
      self.func[0] = 0.
      self.func[-1] = 0.
    if normalize:
      self = self.normalized()
    return None

  def expectation_value(self, operator:'Operator') -> complex:
    """ Return the expectation value of the operator with the wave function: `<self|op(self)>`. """
    return self.scalar_prod(operator(self))

  def variance(self, operator: 'Operator') -> complex:
    """ Return the variance of the operator with the wave function. """
    expval = self.expectation_value(operator)
    dummy = operator(self) - self*expval
    return dummy.scalar_prod(dummy)

  def impose_boundary_condition(self):
    """ Force the wave function to respect the boundary conditions of the grid. """
    if self.grid.bc=='vanishing':
      self.func[0], self.func[-1] = 0., 0.
    elif self.grid.bc=='periodic' or self.grid.bc=='open':
      pass
    else:
      raise NotImplementedError('unknown boundary condition `'+self.bc+'`. Stop!')
    
  def get_observables(self, ops:list):
    """ Return the expectation value and variance for each operator in the list of operators `ops`. """
    obs = []
    for _op in ops:
      obs.append( [self.expectation_value(_op), self.variance(_op)] )
    return np.array(obs)
  
  def evolve(self, tgrid:np.ndarray, op_rhs:'OperatorTD') -> "WavefunctionTD":
    """ 
    Propagate the given initial wavefunction in time, where the rhs of the time evolution equation is `op_rhs`.
    $$
    \partial_t  \Psi = \hat{O}_\text{rhs} \Psi
    $$
    Returns a `WavefunctionTD` object.
    """
    # callable for the ivp solver
    def ipvfunc(t, vec): return op_rhs.sparse_mat(t) * vec

    # solve 
    data = integrate.solve_ivp(fun=ipvfunc, t_span=[tgrid[0], tgrid[-1]], y0=self.func, t_eval=tgrid, method="RK45")
  
    # make the raw output data to class wavefuctions again
    tdwf = WavefunctionTD(self.grid)
    for _i in range(tgrid.size):
      _psi = Wavefunction(self.grid)
      _psi.from_array(data.y[:,_i])
      tdwf.wflist.append(_psi)
    return tdwf
   
  def show(self, file:str=None, absphase:bool=False):
    """ Show a graphical representation of the wave function or save it to file. """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=_wf_figsize)
    ax.set_xlabel('position')
    ax.set_xlim((self.grid.xmin, self.grid.xmax))
    ax.set_ylabel('wave function')
    ax.set_title("")
    if absphase:
      mpl_absphase_plot(fig, ax, self.grid, self.func)
      cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
      mpl_absphase_colorbar(fig, ax, cax=cb_ax)
    else:
      ax.plot(self.grid.points, np.abs(self.func)**2, label='density')
      ax.plot(self.grid.points, np.real(self.func), label='real part')
      ax.plot(self.grid.points, np.imag(self.func), label='imag part')
      fig.legend(loc='upper center', ncol=3)  # , bbox_to_anchor=(1., 1.))
    # show or save
    if file:
      plt.savefig(file)
      plt.close()
    else:
      plt.show()


def GaussianWavePackage(grid: Grid, mu: float = 0, sigma: float = 1, k: float = 1) -> Wavefunction:
  """ 
  Represents a travelling Gaussian wave package. 

  Parameters
  ----------
  grid : Grid
    Grid to evaluate function on.
  mu : float = 0
    Expectation value of the position.
  sigma : float = 1 
    Width of the position distribution (linked to variance).
  k : float=1
    Wave vector/number 

  Returns
  -------
  wf: Wavefunction
    Gaussian wave package wave function.

  Notes
  -----
  The function itself is not normalized, but the wave function gets normalized when calling `.from_func`
  """
  def prep_wf(x): return np.exp(-(x-mu)**2 / (2.0 * sigma**2)) * np.exp(-1j * k * x)
  wf = Wavefunction(grid)
  wf.from_func(prep_wf)
  return wf

  GaussianWavePackage

class WavefunctionTD:
  """ time-dependent wave function class """
  # todo impl algebra
  def __init__(self, grid:Grid):
    self.grid = grid
    self.wflist = []

  def show(self, tgrid:Iterable, file: str = None, pot: Callable[[float, float], float] = None ) -> Union[None, FuncAnimation]:
    """
    Plot the evolution of the wave function
     - optionally plot an additional corresponding time-dependent potential
     - the animation is stored under `file` if present
     - the animation object is returned
    """
    import matplotlib.pyplot as plt
    # disable immediate plotting in interactive mode
    plt.ioff()
    # set a writer
    writer = FFMpegWriter(fps=20)
    # plotting
    if pot:
      fig, (ax, ax2) = plt.subplots(2, 1)
    else:
      fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,wspace=0.05, hspace=0.05)
    wf_min = 0. # min([np.min(np.abs(_psi.func)**2) for _psi in self.wflist])
    wf_max = 1.1 * max([np.max(np.abs(_psi.func)) for _psi in self.wflist])
    # wf_min, wf_max = wf_min-0.01*(wf_max-wf_min), wf_max+0.01*(wf_max-wf_min)
    if pot:
      pot_min = min([pot(_x,_t) for _x in self.grid.points for _t in tgrid])
      pot_max = max([pot(_x,_t) for _x in self.grid.points for _t in tgrid])
      pot_min, pot_max = pot_min-0.01*(pot_max-pot_min), pot_max+0.01*(pot_max-pot_min)

    ax.set_title('evolution of wavefunction')
    ax.set_xlabel('position')
    ax.set_ylabel('density of the wave function')
    ax.set_ylim((wf_min, wf_max))
    ax.set_xlim(self.grid.xmin, self.grid.xmax)
    bar, line_wf = mpl_absphase_plot(fig, ax, self.grid, self.wflist[0].func)
    cb_ax = fig.add_axes([0.83, 0.5, 0.02, 0.4])
    cbar = mpl_absphase_colorbar(fig, ax, cax=cb_ax)
    if pot:
      ax2.set_ylabel('potential')
      ax2.set_xlim((self.grid.xmin, self.grid.xmax))
      ax2.set_ylim((pot_min, pot_max))
      line_pot, = ax2.plot(self.grid.points, [pot(_x, tgrid[0]) for _x in self.grid.points])
      line_pot.set_xdata(self.grid.points)
    text = ax.text(0.8, 0.9, '', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    def animate(i):
      _t = float(tgrid[i])
      text.set_text('time='+'{:.2f}'.format(_t))
      mpl_absphase_plot_update(fig, ax, self.grid, self.wflist[i].func, bar, line_wf)
      if pot:
        line_pot.set_ydata(np.array([pot(_t, _x) for _x in self.grid.points]))
        line_pot.set_ydata([pot(_x,_t) for _x in self.grid.points])
        return line_wf, line_pot, text,
      else:
        return line_wf, text,

    ani = FuncAnimation(fig=fig, func=animate, frames=range(len(self.wflist)), interval=1000./24.)
    if file: ani.save(file, writer=writer)
    # close the figure
    plt.close()
    # enable plotting again in interactive mode
    plt.ion()
    return ani
