# Quantum mechanics in one dimension

A package for calculation and visualization of stationary or time-dependent basic one-dimensional (undergrade) problems. In the stationary scenario problems like potential steps or barriers, harmonic oscillators, etc. can be easily solved. In the time-dependent case tunneling can be calculated and visualized. Focus is on visualization and intuitive handling of the basic entities of qm: operators, wave functions, eigen systems, measurements... Speed is fine, but not prioritized.

The below example shows the time evolution of a user-defined initial wave function (a linear combination of few low energy eigenstates) under the action of a time dependend potential.

![tdmwe_wavefunc](https://user-images.githubusercontent.com/98354510/200693510-e6dd1e5f-555e-4e88-88f1-e22d37e62e00.gif)

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
eigsys = Eigensystem(op_hamilton, num=10)
eigsys.show('README_eigensystem.png')
```
Resulting in the plot:

![README_eigensystem](https://user-images.githubusercontent.com/98354510/200693955-50011bda-7fb3-427f-8e5b-b7a119d8e524.png)




## How-To
The classes to describe the quantum mechanical entities are kept simple. Useability and readability of the code are given higher priority than speed. Here are some examples of the API.
### wave functions
Wave functions are represented as a vector (`np.ndarray`) of function values on the given point-grid and take complex values in general. In explicitly real and stationary systems wave functions can also be real. New instances of wave functions can be generated and set with 
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
Time dependent wave functions `WavefunctionTD` are a list of `Wavefunction` objects.
### operators
Operators act as matrices on the wave functions. Sparse matrix representations are used to store the elements for fast calculations. 

Operators act on wave functions via their `__call__` method

```python
op_acting_on_wf = op(wf)
```

Two kinds of operators are discriminated: `OperatorConst` and `OperatorTD`.
#### Time-constant operators
Time-constant Operators get instanciated with dependency on a given `grid`.
```python
 op = OperatorConst(grid)
```
To set the matrix elements you can either set the diagonal of a local operator (with a callable `func`)
```python
 op.local(func(grid.points))
```
set derivatives
```python
  op.first_deriv()
  op.second_deriv()
```
or use the build-in arithmetic operators (`+-*`) to make more complex operators from basic ones.
```python
  op = _op * (-0.5/qsys.mass)
```
Setting the matrix elements all by yourself is also perfectly fine. Here `mat` is a size-compatible matrix with arbitrary elements.
```python
  op.from_matrix(mat)
```
To fasten coding there are predefined operators that can be instanciated with reference to the `QMSystem`
```python
op_identity = IdentityOp(qsys.grid)
op_position = PositionOp(qsys.grid)
op_momentum = MomentumOp(qsys.grid)
op_hamilton = HamiltonOp(qsys) # takes the potential directly from `qsys`
# as well as: ZeroOp(grid:Grid), GradientOp(grid:Grid), LaplaceOp(grid:Grid), StatPotentialOp(qsys:QMSystem), KineticOp(qsys:QMSystem), ...
```
When the operators have been defined via any of the above methods, the operator can be cast in sparse representation with 
```python
op.finalize()
```
Now the representation is more efficient in memory and runtime.
#### Time-dependent operators
The representation of time-dependent operators is implemented as follows:
$$
O_t = O_0 + \sum_{k=1}^N f_k(x, t) O_k 
$$ 
Any time-dependent operator consists of a single time-constant operator $O_0$ and a series of products
$$
f_k(x, t) O_k 
$$
where $f_k(x, t)$ is a function of time and space and $O_k$ is a `OperatorConst` operator. Addition, substraction and negation are implemented as class operations, but multiplication (sequential application, concatenation) is only allowed with scalars. This class should be handled with some slight care, since for example the repeated addition of a simple operator is not automatically understood as multiplication with an integer scalar, thus consumes memory and increases runtime. 

The main use case for time-dependent operators are td systems (with td potentials and thus td Hamiltonians). 

### eigen system
The (fraction of the) eigensystem of an operator can be found easily with the `Eigensystem` class:
```python
es = Eigensystem(op, num=5):
```
`Eigensystem` has plotting and evaluation routines implemented.
### measurements
To link the mathematical framework of quantum mechanics with the experiment the `Measure` class simulates measurements on wavefunctions. To perform a measurement on a wave function first instantiate the measurement with reference to the observable/operator you want to measure and specify the number of eigen states up to which the observable can be resolved. For example: When measuring the energy using the Hamiltonian operator with `num_states=100`, the measurement routines will decompose the wave function to be measured into the first 100 (energy-) eigen states and draw eigen energies with the probabilities of those states. 
```python
m = Measure(op, num_states=100):
```
Next, specify the wave function `prepared_wf` to be measured. Then the repeated measurement of `num_obs=1000` single measurements happens in 
```python
m(prepared_wf, num_obs=1000)
```
To plot the timeline and hiostogram of the measurement, use
```python
m.show(file='measure.png')
```
Note: When the prepared wave function does not lie in the span of the eigen system of `m` (or the fraction of it, `num_states=100`) the measurement will be recorded as out-of-bounds. 
## remarks
Hartree atomic units `hbar = m = e = 1/4pi epsilon_0` are used in the whole code.
