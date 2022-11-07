# Quantum mechanics in one dimension

A package for calculation and visualization of stationary or time-dependent basic one-dimensional (undergrade) problems. In the stationary scenario problems like potential steps or barriers, harmonic oscillators, etc. can be easily solved. In the time-dependent case tunneling can be calculated and visualized. Focus is on visualization and intuitive handling of the basic entities of qm: operators, wave functions, eigen systems, measurements... Speed is fine, but not prioritized.

## features 
- classes for the main objects in quantum mechanics: wave functions and operators
- Handles eigensystems, observables, time evolution
- predefined and customizable operators
- easy plotting
## examples
`mwe.ipynb` implements a minimal working example for a stationary quantum system, `mwe_td.ipynb` for the time-dependend case respectivley. `tutorial_stat.ipynb` shows in detail steps for customization.

API as easy as 
```python
from qm1.grid import UniformGrid
from qm1.qmsystem import QMSystem, BarrierPot
from qm1.operators import HamiltonOp
from qm1.eigensystem import Eigensystem
grid = UniformGrid(boundary_condition="vanishing", xmin=-20., xmax=20., num=250)
potential = BarrierPot(xstart=-1., xstop=+2., vstep=-1.)
qsys = QMSystem(grid, potential)
op_hamilton = HamiltonOp(qsys)
eigsys = Eigensystem(operator=op_hamilton, num=10)
eigsys.show('README_eigensystem.png')
```
## remarks
Hartree atomic units `hbar = m = e = 1/4pi epsilon_0` are used in the whole code.


## How-To
The classes to describe the quantum mechanical entities are kept simple. Useability and readability of the code are given higher priority than speed. Here are some examples of the API.
### wave functions
Wave functions are represented as a vector (`np.ndarray`) of function values on the given point-grid and take complex values in general. In explicitly real and stationary systems, wave functions are also real. Operators act as matrices on the wave functions. New instances of wave functions can be generated and set with 
```python 
wf = Wavefunction(grid)
func = lambda x: np.exp(-x**2)
wf.from_func(func)
```
Wave functions can also be set via an existing one-dimensional `np.ndarray` with the `from_array` function.
Visualization is available throught
```python 
wf.show(file='wf.png')
```
Special types of wave functions can be generated with built-ins, e.g.
```python
wf = GaussianWavePackage(grid, mu=0, sigma=1, k=0.1)
```
When given a stationary operator `op` the expectation value and variance can be computed with
```python
wf.expectation_value(op)
wf.variance(op)
```
or even for a bunch of operators, e.g.
```python
obs = wf.get_observables([op_identity, op_position, op_momentum, op_hamilton])
```

### operators
todo
### eigen system
todo
### measurements
todo