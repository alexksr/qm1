import numpy as np
import matplotlib.pyplot as plt
from qm1.operators import OperatorConst
from qm1.wavefunction import Wavefunction
from typing import List

class Eigensystem:
  """Handles calculation and storage of the eigen system of an operator."""
  def __init__(self, operator: OperatorConst, num:int=10):
    """
    Calculates and stores the eigensystem (with `num` eigen states) of `operator`.
    
    Parameters
    ----------
    ``operator : OperatorConst``
      (Hermitian) operator to calc eigensystem for.
    ``num : int``
      Number of eigen states (and values) to compute.
  """
    self.num = num
    self.op = operator
    self.grid = operator.grid

    # get a generic initial guess wf
    init_wf = Wavefunction(self.grid)
    length = self.grid.xmax-self.grid.xmin
    mu, sigma = self.grid.xmin + 0.5*length, 0.1
    def func(x): return np.sin((x-mu)/length*np.pi) * np.exp(-0.5*((x-mu)/length/sigma)**2)
    init_wf.from_func(func)

    # solve the evp
    self.eigvals, self.eigstates = operator.eigen_system(self.num, init_wf)

    # normalize the eigen states
    for _i, _ in enumerate(self.eigstates):
      self.eigstates[_i] = self.eigstates[_i].normalized()

  def decompose(self, wavefunc: Wavefunction):
    """ 
    Decompose a wave function into the basis of the eigen system. 
    Return expansion coefficients and rest term.

    Parameters
    ----------
    ``wavefunc: Wavefunction``
      wave function to decompose 

    Returns
    -------
    ``coefficients:``
      List of floats or complex coeficients (with length given by ``self.num``)
    ``rest: float``
      Missing ``probability`` to fullfill normalization.
    """
    coefficients = np.zeros((self.num))
    for _i in range(self.num):
      coefficients[_i] = wavefunc.scalar_prod(self.eigstates[_i])
    p_coverage = np.sum(np.abs(coefficients)**2)
    if p_coverage>1.+1e-5: 
      raise ValueError('Probabilities sum to {:.4f} (more than one). corrupt eigensystem or low spatial resolution?'.format(p_coverage))
    rest = 1.-np.sum(np.abs(coefficients)**2)
    return coefficients, rest

  def show(self, file:str=None, state_range:tuple=None, op_pot:OperatorConst=None):
    """
    Plot the eigensystem.
    Returns a plot of the eigen vector and corresponding eigen values.

    Parameters
    ----------
    ``file: str``
      File to save the figure to. When present the figure is saved to file, otherwise (if 'None') the figure will be displayed immediately.
    ``op_pot: OperatorConst``
      When present plot the potential operator (or any local operator) next to the eigensystem.
    ``state_range: tuple``
      When present plot only the specified range of states. For example `state_range=(2,4)`
    """
    # handle range
    if state_range:
      _state_range = state_range
    else:
      _state_range = (0, len(self.eigstates))
    if not op_pot is None:
      fig, (ax0, ax1)=plt.subplots(2, 1, gridspec_kw = {'height_ratios': [4, 2]}, sharex=True)
      fig.subplots_adjust(hspace = .0)
    else:
      fig, ax0 = plt.subplots()
    colors = [plt.cm.tab10(i) for i in range(self.num)]
    alphas = [((self.num-0.5*i ) / self.num)**2. for i in range(self.num)]
    ax0.set_title('eigen system')
    ax0.set_ylabel('eigen states / wave functions')
    ax0.set_xlabel('position')
    ax0.set_xlim((self.grid.xmin, self.grid.xmax))
    for _i, (_eigs, _col, _a) in enumerate(zip(self.eigstates, colors, alphas)):
      if _state_range[0] <= _i <= _state_range[1]:
        ax0.plot(self.grid.points, _eigs.func, label='state '+str(_i), alpha=_a, color=_col)
    ax0.legend(loc='center right')
    axt = ax0.twinx()
    axt.set_ylabel('operator eigenvalues')
    for _eigval, _col, _a in zip(self.eigvals, colors, alphas):
      if _state_range[0] <= _i <= _state_range[1]:
        axt.axhline(_eigval, color=_col, alpha=_a, ls='--')
    if not op_pot is None:
      ax1.set_xlim((self.grid.xmin, self.grid.xmax))
      ax1.plot(self.grid.points, op_pot.get_diag(), "k")
      ax1.set_xlabel('position')
      ax1.set_ylabel('potential')
    if file:
      plt.savefig(file)
      plt.close()
    else:
      plt.show()


  def get_observables(self, ops: List[OperatorConst]) -> np.ndarray:
    """ 
    Return the expectation value and variance for each operator in the list of operators `ops` for each eigenstate in the eigensystem.
    
    Parameters
    ----------
    ``ops: List[OperatorConst]``
      List of operators to eval across the eigensystem.

    Returns
    -------
    ``observations : np.ndarray``
      Array of obervations ordered like obervations[eigenstate][operator][exp,var]
    """
    observations = []
    for _eigstate in self.eigstates:
      observations.append(_eigstate.get_observables(ops))
    observations =np.array(observations)
    return observations
