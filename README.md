# Quantum mechanics in one dimension

A small package for calculation and visualization of stationary or time-dependent basic one-dimensional (undergrade) problems. In the stationary scenario problems like potential steps or barriers, harmonic oscillators, etc. can be easily solved. In the time-dependent case tunneling can be calculated and visualized.

## features 
- classes for the main objects in quantum mechanics: wave functions and operators
- Handles eigensystems, observables, time evolution
- predefined and customizable operators
- easy plotting
## exmaples
`mwe.ipynb` implements a minimal working example for a stationary quantum system, `mwe_td.ipynb` for the time-dependend case respectivley.
API as easy as 
```python
from qm1.grid import *
from qm1.qmsystem import *  
from qm1.operators import *
from qm1.eigensystem import Eigensystem
grid = UniformGrid(boundary_condition="vanishing", xmin=-20., xmax=20., num=250)
potential = BarrierPot(xstart=-1., xstop=+2., vstep=-1.)
qsys = QMSystem(potential, grid)
op_hamilton = HamiltonOp(qsys)
eigsys = Eigensystem(qsys=qsys, operator=op_hamilton)
eigsys.show('eigensystem.png')
```
## remarks
Hartree atomic units `hbar = m = e = 1/4pi epsilon_0` are used in hte hwole code.