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


def mpl_absphase_colorbar(fig, ax, cax):
  cbar = fig.colorbar(mappable=_wf_anim_cm_mappable, ax=ax, cax=cax, ticks=_wf_anim_cm_phase_ticks, fraction=0.046, pad=0.04)
  cbar.set_ticklabels(_wf_anim_cm_phase_labels)
  return cbar

def mpl_absphase_plot(fig, ax, grid: Grid, func: np.ndarray):
  colors = _wf_anim_cmap(np.angle(func)/2./np.pi + 0.5)
  bar = ax.bar(grid.points, np.abs(func), color=colors, width=grid.dx)
  line,  = ax.plot(grid.points, np.abs(func), c='black', lw=3)
  # cbar = fig.colorbar(mappable=_wf_anim_cm_mappable, ax=ax, ticks=_wf_anim_cm_phase_ticks, fraction=0.046, pad=0.04)
  # cbar.set_ticklabels(_wf_anim_cm_phase_labels)

  return  bar, line#, cbar


# Line2D(xdata, ydata, *, linewidth=None, linestyle=None, color=None, gapcolor=None, marker=None, markersize=None, markeredgewidth=None, markeredgecolor=None, markerfacecolor=None, markerfacecoloralt='none', fillstyle=None, antialiased=None, dash_capstyle=None, solid_capstyle=None, dash_joinstyle=None, solid_joinstyle=None, pickradius=5, drawstyle=None, markevery=None, **kwargs)
# np.abs(func)**2
# bar < class 'matplotlib.container.BarContainer' >
# line < class 'list' >
# cbar < class 'matplotlib.colorbar.Colorbar' >


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

  #   for 
    
    
  


  # def animated_barplot():
  #   # http://www.scipy.org/Cookbook/Matplotlib/Animations
  #   mu, sigma = 100, 15
  #   N = 4
  #   x = mu + sigma*np.random.randn(N)
  #   rects = plt.bar(range(N), x,  align='center')
  #   for i in range(50):
  #       x = mu + sigma*np.random.randn(N)
  #       for rect, h in zip(rects, x):
  #           rect.set_height(h)


  # fig = plt.figure()
  # win = fig.canvas.manager.window
  # win.after(100, animated_barplot)
  # plt.show()



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

  def __mul__(self, other:Union[float, int, complex]) -> 'Wavefunction':
    """ multiply a Wavefunction by a constant """
    result = Wavefunction(self.grid)
    result.func = other * self.func
    return result
  
  def __rmul__(self, other: Union[float, int, complex]) -> 'Wavefunction':
    """ multiply a Wavefunction by a constant """
    result = Wavefunction(self.grid)
    result.func = self.func * other
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

  def from_array(self, func: np.ndarray, normalize:bool=True):
    """ set the wavefunction to the given array of values `func` """
    self.func = func
    if self.grid.bc == 'vanishing':
      self.func[0] = 0.
      self.func[-1] = 0.
    if normalize: 
      self = self.normalized()
    return None

  def from_func(self, func:Callable[[float], Union[float,complex]] , normalize: bool = True):
    """ 
    Set the wavefunction by evaluating the given function on the grid.
    Automatically do a normalization.
    """
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
  
  def evolve(self, tgrid:np.ndarray, op_rhs:'OperatorTD') -> "WavefunctionTD":
    """ 
    propagate the wavefunction in time 
    - return a `WavefunctionTD` object
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
    """
    Save a graphical representation of the wave function to file.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import colors as mcolors
    figsize = (12, 7)
    fig, ax = plt.subplots(figsize=figsize)
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
      
    if file:
      plt.savefig(file)
      plt.close()
    else:
      plt.show()


def GaussianWavePackage(grid: Grid, mu: float = 0, sigma: float = 1, k: float=1):
  def prep_wf(x): return np.exp(-(x-mu)**2 / (2.0 * sigma**2)) * np.exp(-1j * k * x)
  wf = Wavefunction(grid)
  wf.from_func(prep_wf)
  return wf

class WavefunctionTD:
  """ wave function with time dependence """
  # todo impl algebra
  def __init__(self, grid:Grid):
    self.grid = grid
    self.wflist = []

  def show(self, tgrid:Iterable, file: str = None, pot: Callable[[float, float], float] = None ) -> Union[None, FuncAnimation]:
    """Plot the evolution of the wave function
     - optionally plot an additional corresponding time dependend potential
     - the animation is stored under `file` if present
     - the animation object is returned
    """
    import matplotlib.pyplot as plt
    # disable immediate plotting in interactive mode
    plt.ioff()
    # set a writer
    writer = FFMpegWriter(fps=24)
    # plotting

    if pot:
      fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    else:
      fig, ax = plt.subplots(1, 1, figsize=(10, 10))
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
    # cbar_ax = fig.add_axes([0.95, 0.55, 0.025, 0.4])
    # cbar = fig.colorbar(_wf_anim_cm_mappable, cax=cbar_ax)
    cb_ax = fig.add_axes([0.83, 0.5, 0.02, 0.4])
    # fig.colorbar(_wf_anim_cm_mappable, cax=cb_ax)
    cbar = mpl_absphase_colorbar(fig, ax, cax=cb_ax)
    if pot:
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

  def show2(self, tgrid: Iterable, file: str = None, pot: Callable[[float, float], float] = None) -> Union[None, FuncAnimation]:
    """Plot the evolution of the wave function
     - optionally plot an additional corresponding time dependend potential
     - the animation is stored under `file` if present
     - the animation object is returned
    """
    import matplotlib.pyplot as plt
    # disable immediate plotting in interactive mode
    plt.ioff()
    # set a writer
    writer = FFMpegWriter(fps=24)
    # plotting
    _xgrid = self.grid.points
    if pot:
      fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    else:
      fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    y1min = min([np.min(np.abs(_psi.func)**2) for _psi in self.wflist])
    y1max = max([np.max(np.abs(_psi.func)**2) for _psi in self.wflist])
    y1min, y1max = y1min-0.01*(y1max-y1min), y1max+0.01*(y1max-y1min)
    if pot:
      y2min = min([pot(_x, _t) for _x in _xgrid for _t in tgrid])
      y2max = max([pot(_x, _t) for _x in _xgrid for _t in tgrid])
      y2min, y2max = y2min-0.01*(y2max-y2min), y2max+0.01*(y2max-y2min)

    ax.set_title('evolution of wavefunction')
    ax.set_xlabel('position')
    ax.set_ylabel('density of the wave function')
    ax.set_ylim((y1min, y1max))
    ax.set_xlim(min(_xgrid), max(_xgrid))
    line, = ax.plot([], [])
    if pot:
      ax2.set_xlim(min(_xgrid), max(_xgrid))
      ax2.set_ylim((y2min, y2max))
      line2, = ax2.plot([], [])
    text = ax.text(0.8, 0.9, '', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    def animate(i):
      _t = float(tgrid[i])
      text.set_text('time='+'{:.2f}'.format(_t))
      if i < 1:
        line.set_xdata(_xgrid)
      line.set_ydata(np.abs(self.wflist[i].func)**2)
      if pot:
        if i < 1:
          line2.set_xdata(_xgrid)
        line2.set_ydata(np.array([pot(_t, _x) for _x in _xgrid]))
        line2.set_ydata([pot(_x, _t) for _x in _xgrid])
        return line, line2, text,
      else:
        return line, text,

    ani = FuncAnimation(fig=fig, func=animate, frames=range(len(self.wflist)), interval=1000./24.)
    if file:
      ani.save(file, writer=writer)
    # close the figure
    plt.close()
    # enable plotting again in interactive mode
    plt.ion()
    return ani


##########################################################################################################from matplotlib.animation import FuncAnimation, FFMpegWriter

  # def show(self, tgrid: Iterable, file: str = None, pot: Callable[[float, float], float] = None, absphase: bool = False) -> Union[None, FuncAnimation]:
  #   """Plot the evolution of the wave function
  #    - optionally plot an additional corresponding time dependend potential
  #    - the animation is stored under `file` if present
  #    - the animation object is returned
  #   """
  #   import matplotlib.pyplot as plt
  #   from matplotlib import colors as mcolors

  #   # disable immediate plotting in interactive mode
  #   plt.ioff()
  #   # set a writer
  #   writer = FFMpegWriter(fps=24)
  #   # plotting
  #   if pot:
  #     fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10))
  #   else:
  #     fig, ax = plt.subplots(1, 1, figsize=(10, 10))

  #   y1min = min([np.min(np.abs(_psi.func)**2) for _psi in self.wflist])
  #   y1max = max([np.max(np.abs(_psi.func)**2) for _psi in self.wflist])
  #   y1min, y1max = y1min-0.01*(y1max-y1min), y1max+0.01*(y1max-y1min)

  #   if pot:
  #     y2min = min([pot(_x, _t) for _x in self.grid.points for _t in tgrid])
  #     y2max = max([pot(_x, _t) for _x in self.grid.points for _t in tgrid])
  #     y2min, y2max = y2min-0.01*(y2max-y2min), y2max+0.01*(y2max-y2min)

  #   ax.set_title('evolution of wavefunction')
  #   ax.set_xlabel('position')
  #   ax.set_ylabel('density of the wave function')
  #   ax.set_ylim((y1min, y1max))
  #   ax.set_xlim(self.grid.xmin, self.grid.xmax)
  #   line_wf, = ax.plot([], [])
  #   bar_wf,  = ax.bar(self.grid.points, np.abs(self.wflist[0].func)**2, color=_wf_anim_cmap(np.angle(self.wflist[0].func)), width=self.grid.dx)
  #   plot_wf,  = ax.plot(self.grid.points, np.abs(self.wflist[0].func)**2, c='black', lw=3)

  #   if pot:
  #     ax2.set_xlim(self.grid.xmin, self.grid.xmax)
  #     ax2.set_ylim((y2min, y2max))
  #     line_pot, = ax2.plot([], [])
  #   text = ax.text(0.8, 0.9, '', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

  #   def animate(i):

  #     _t = float(tgrid[i])
  #     text.set_text('time='+'{:.2f}'.format(_t))
  #     if i < 1:
  #       line_wf.set_xdata(self.grid.points)

  #     cphase = np.angle(self.wflist[i].func)
  #     colors = _wf_anim_cmap(cphase)
  #     mappable = cm.ScalarMappable(norm=_wf_anim_cm_norm, cmap=_wf_anim_cmap)
  #     cbar = fig.colorbar(mappable=mappable, ax=ax, ticks=_wf_anim_cm_phase_ticks)  # , orientation='vertical')
  #     cbar.set_ticklabels(_wf_anim_cm_phase_labels)

  #     bar_wf.set_ydata(np.abs(self.wflist[i].func)**2)
  #     line_wf.set_facecolor(colors)

  #     if pot:
  #       if i < 1:
  #         line_pot.set_xdata(self.grid.points)
  #       line_pot.set_ydata(np.array([pot(_t, _x) for _x in self.grid.points]))
  #       line_pot.set_ydata([pot(_x, _t) for _x in self.grid.points])
  #       return line_wf, line_pot, text,
  #     else:
  #       return line_wf, text,

  #   ani = FuncAnimation(fig=fig, func=animate, frames=range(len(self.wflist)), interval=1000./24.)
  #   if file:
  #     ani.save(file, writer=writer)
  #   # close the figure
  #   plt.close()
  #   # enable plotting again in interactive mode
  #   plt.ion()
  #   return ani
