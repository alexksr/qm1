import sys
sys.path.append("../qm1")
from qm1.operators import *
from qm1.grid import *
from qm1.operators import ZeroOp
from qm1.qmsystem import *
from qm1.eigensystem import Eigensystem
from qm1.basics import nmetric

# bounds and accuracy of grid are tailored to be just enough to pass the tests for the observables
grid = UniformGrid(boundary_condition="vanishing", xmin=-.5, xmax=+.5, num=3000)
stat_pot = ConstPot()
qsys = QMSystem(grid=grid, stat_pot=stat_pot, mass=1.)
op_hamilton = HamiltonOp(qsys)  # takes the potential directly from `qsys`
op_identity = IdentityOp(qsys.grid)
op_position = PositionOp(qsys.grid)
eigsys = Eigensystem(operator=op_hamilton)
obs = eigsys.get_observables([op_identity, op_position, op_hamilton])
for _istate in range(len(eigsys.eigstates)):
  assert np.abs(obs[_istate][0][0]-1.) < 1E-8, 'when checking harmonic oscillator: incorrect norm of state '+str(_istate)+': '+str(obs[_istate][0][0])
  assert np.abs(obs[_istate][1][0]-0.) < 1E-8, 'when checking harmonic oscillator: incorrect exp-val for position of state '+str(_istate)+': '+str(obs[_istate][1][0])
  assert nmetric(obs[_istate][2][0], 0.5*np.pi**2*(_istate+1)**2) < 1E-3, 'when checking harmonic oscillator: incorrect exp-val for energy of state ' + \
      str(_istate)+': '+str(obs[_istate][2][0])+' vs. '+str(0.5*np.pi**2*(_istate+1)**2)
