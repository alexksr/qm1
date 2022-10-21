import sys
sys.path.append("../qm1")
from qm1.wavefunction import GaussianWavePackage
from qm1.operators import *
from qm1.grid import *
from qm1.operators import ZeroOp
from qm1.qmsystem import *
from qm1.eigensystem import Eigensystem

# bounds and accuracy of grid are tailored to be just enough to pass the tests for the observables
grid = UniformGrid(boundary_condition="vanishing", xmin=-15., xmax=15., num=3000)
stat_pot = HarmonicPot(omega=1.)
qsys = QMSystem(stat_pot=stat_pot, grid=grid, mass=1.)
op_hamilton = HamiltonOp(qsys)  # takes the potential directly from `qsys`
op_identity = IdentityOp(qsys.grid)
op_position = PositionOp(qsys.grid)
eigsys = Eigensystem(qsys=qsys, operator=op_hamilton)
obs = eigsys.get_observables([op_identity, op_position, op_hamilton])
for _istate in range(len(eigsys.eigstates)):
  assert np.abs(obs[_istate][0][0]-1.) < 1E-8, 'when checking harmonic oscillator: incorrect norm of state '+str(_istate)+': '+str(obs[_istate][0][0])
  assert np.abs(obs[_istate][1][0]-0.) < 1E-8, 'when checking harmonic oscillator: incorrect exp-val for position of state '+str(_istate)+': '+str(obs[_istate][1][0])
  assert np.abs(obs[_istate][2][0]-(0.5+_istate)) < 1E-3, 'when checking harmonic oscillator: incorrect exp-val for energy of state '+str(_istate)+': '+str(obs[_istate][2][0])
