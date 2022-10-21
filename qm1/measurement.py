from difflib import context_diff
import numpy as np
import matplotlib.pyplot as plt
from qm1.operators import OperatorConst
from qm1.wavefunction import Wavefunction
from qm1.eigensystem import Eigensystem

class Measure:
  def __init__(self, op: OperatorConst, num_states:int=10):
    self.op = op
    self.num_states = num_states
    # get the eigensystem
    self.eigsys = Eigensystem(operator=op, num=self.num_states)

  def __call__(self, wf: Wavefunction, num_obs: int = 1000) -> np.ndarray:
    """
    Simulate a series of experiments to measure operator `self.op` in the state given by wave function `wf`.
        
    Parameters
    ----------
    wf : Wavefunction
        wave function to measure
    num_obs : int, optional
        number of experiments/measurements to simulate, defaults to 1000

    Returns
    -------
    self.measured_eigvals : ndarray
        An array containing the measured values. These are also stored in the `Measure` class.


    Notes
    -----
    For every eigenstate in `self.eigsys` the overlap with the given `wf` is calculated and only eigenvalues in the eigensystem can be measured. 
    When the probabilities to measure a eigenvalue in `self.eigsys` do not sum to one, a `None` value is drawn with the "remaining" probability.
    """
    self.coeffs, self.rest = self.eigsys.decompose(wf)
    p_coverage = np.sum(np.abs(self.coeffs)**2)
    if p_coverage < .5:
      print('WARNING: Decomposition of the wave function in the eigensystem retrieves only {:.2f} of the coefficients. Try to take more states into account.'.format(p_coverage))
    rng = np.random.default_rng()
    self.measured_eigvals = rng.choice(self.eigsys.eigvals+[None], size=num_obs, p=list(np.abs(self.coeffs)**2)+[self.rest])
    self.actual_meas_eigvals = [_m for _m in self.measured_eigvals if _m is not None]
    self.num_none = len(self.measured_eigvals)-len(self.actual_meas_eigvals)
    return self.measured_eigvals

  def show(self, file:str=None):
    """
    Visualize the measurement:
     - plot time series of measurement outcomes
     - plot histogram with expectation value and standard deviation
     - when `file` is present, `savefig` to file
    """
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10,10))
    ax0.set_title('measurement outcome')
    ax0.set_xlabel('measurements')
    ax0.set_ylabel('observed outcome')
    ax0.scatter(np.arange(len(self.actual_meas_eigvals)), self.actual_meas_eigvals)
    ax0.text(0.6, 0.9, r'number of observations out-of-bounds='+'{:.4f}'.format(self.num_none), horizontalalignment='left', verticalalignment='center', transform=ax0.transAxes)
    ax1.set_title('histogram of outcome')
    ax1.set_xlabel('observed outcome')
    ax1.set_ylabel('probability')
    # add some lines to the histogram to vis exp and var
    counts, bins = np.histogram(self.actual_meas_eigvals, bins=len(self.actual_meas_eigvals), density=True)
    ymid = 0.5*(min(counts)+max(counts))
    exp = np.mean(self.actual_meas_eigvals)
    sigma = np.sqrt(np.var(self.actual_meas_eigvals))
    ax1.axvline(x=exp, color='red')
    ax1.axvspan(xmin=exp-sigma, xmax=exp+sigma, facecolor='0.9')
    ax1.bar(bins[:-1], counts)
    ax1.hlines(y=ymid, xmin=exp-sigma, xmax=exp+sigma)
    ax1.text(0.8, 0.9, 'mean: '+'{:.4f}'.format(exp)+'\nstd-dev: '+'{:.4f}'.format(sigma), horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    if file:
      plt.savefig(file)
      plt.close()
    else:
      plt.show()
    return

  
