from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
from qm1.operators import OperatorConst
from qm1.qmsystem import QMSystem
from qm1.wavefunction import Wavefunction

class Eigensystem:
  def __init__(self, operator: OperatorConst, num:int=10):
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
    """ decompose a wavefunction into a spectral basis with coefficients and rest term"""
    coefficients = np.zeros((self.num))
    for _i in range(self.num):
      coefficients[_i] = wavefunc.scalar_prod(self.eigstates[_i])
    p_coverage = np.sum(np.abs(coefficients)**2)
    if p_coverage>1.: 
      raise ValueError('Probabilities sum to {:.4f} (more than one). corrupt eigensystem or low spatial resolution?'.format(p_coverage))
    rest = 1.-np.sum(np.abs(coefficients)**2)
    return coefficients, rest

  def show(self, file:str=None, op_pot:OperatorConst=None):
    if not op_pot is None:
      fig, (ax0, ax1)=plt.subplots(2, 1, figsize = (15, 5), gridspec_kw = {'height_ratios': [4, 2]}, sharex=True)
      fig.subplots_adjust(hspace = .0)
    else:
      fig, ax0 = plt.subplots(figsize=(15, 7))
    colors = [plt.cm.tab10(i) for i in range(self.num)]
    alphas = [((self.num-0.5*i ) / self.num)**2. for i in range(self.num)]
    ax0.set_title('eigenstates of the hamiltonian')
    ax0.set_ylabel('eigen states / wave functions')
    ax0.set_xlabel('position')
    ax0.set_xlim((self.grid.xmin, self.grid.xmax))
    for _i, (_eigs, _col, _a) in enumerate(zip(self.eigstates, colors, alphas)):
      ax0.plot(self.grid.points, _eigs.func, label='state '+str(_i), alpha=_a, color=_col)
    ax0.legend(loc='center right')
    axt = ax0.twinx()
    axt.set_ylabel('operator eigenvalues')
    for _eigval, _col, _a in zip(self.eigvals, colors, alphas):
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


  def get_observables(self, ops: list):
    """ 
    Return the expectation value and variance for each operator in the list of operators `ops` for each eigenstate in the eigensystem
    """
    obs = []
    for _eigstate in self.eigstates:
      obs.append(_eigstate.get_observables(ops))
    return  np.array(obs)