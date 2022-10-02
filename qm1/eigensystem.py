import numpy as np
import matplotlib.pyplot as plt
from qm1.operators import OperatorConst
from qm1.qmsystem import QMSystem
from qm1.wavefunction import Wavefunction

class Eigensystem:
  def __init__(self, qsys:QMSystem, operator: OperatorConst, num:int=10):
    self.num = num
    self.op = operator
    self.grid = operator.grid
    self.qsys = qsys

    # get a generic initial guess wf
    init_wf = Wavefunction(self.grid)
    length = self.grid.xmax-self.grid.xmin
    mu, sigma = self.grid.xmin + 0.5*length, 0.1
    def func(x): return np.sin((x-mu)/length*np.pi) * np.exp(-0.5*((x-mu)/length/sigma)**2)
    init_wf.set_via_func(func)

    # solve the evp
    self.eigvals, self.eigstates = operator.eigen_system(self.num, init_wf)

    # normalize the eigen states
    for _i, _ in enumerate(self.eigstates):
      self.eigstates[_i] = self.eigstates[_i].normalized()

  def decompose(self, wavefunc: Wavefunction):
    """ decompose a wavefunction into a spectral basis with coefficients and rest term"""
    coefficients = []
    for _i in range(self.num):
      coefficients.append(np.abs(wavefunc.scalar_prod(self.eigstates[_i]))**2.)
    coefficients = np.array(coefficients)
    rest = 1.-np.sum(coefficients)
    return coefficients, rest

  def show(self, file):
    fig, ax1 = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(right=0.6)
    colors = [plt.cm.tab10(i) for i in range(self.num)]
    alphas = [((self.num-i) / self.num)**2. for i in range(self.num)]
    ax1.set_title('eigenstates of the hamiltonian')
    ax1.set_xlabel('position')
    ax1.set_ylabel('wave function')
    ax1.set_xlim((self.grid.xmin, self.grid.xmax))
    for _i, (_eigs, _col, _a) in enumerate(zip(self.eigstates, colors, alphas)):
      ax1.plot(self.grid.points, _eigs.func, label='state '+str(_i), alpha=_a, color=_col)
    ax1.legend(loc='center right')
    ax2 = ax1.twinx()
    ax2.set_ylabel('operator eigenvalues')
    ax2.plot(self.grid.points, self.qsys.pot, "k")
    for _eigval, _col, _a in zip(self.eigvals, colors, alphas):
      ax2.axhline(_eigval, color=_col, alpha=_a, ls='--')
    plt.savefig(file)
    plt.close()
