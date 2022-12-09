# TODO
## Recent
- [ ] impl tdwf algebra
- [ ] non uniform grids
- [ ] add a check to make sure the local ffmpeg installation can animate gifs correctly (codec, mpl.anim.writers)
- [ ] impl eigenfunctions of the box potential as built-in WFs
- [ ] make a class for potentials (const, td)
- [ ] in tdwf.show: make layout nicer
- [ ] add a sinksource module with particle generation/destruction functionality
# BACKLOG
- [ ] make pip package
# DONE
- [x] add observables functionality to return the exp-vals or variances of a wave func wrt some operator
- [x] time evolution with vanishing bc not working: inconsistent matrices with the first and last row not beeing zero
- [x] make wavefunction class compatible with time-dependence or add a WavefunctionTD class
- [x] OperatorTD works
- [x] auto generate plots via a `show` method
- [x] easy-to-use predefine operators
- [x] add plot style for complex wf: abs and phase with animation combatibility
- [x] add MWEs
- [x] make `gif` working in outputs of videos (maybe use celluloid)
- [x] add measurement functionality and visualization
- [x] `show` not working for td-wf with td-potential
- [x] in the `show` routines: return animations by default to window or notebook, plot static graphics. when `file` is present also use `savefig`
- [x] add plot style for complex wf: abs and phase
- [x] add predefines Hamiltonian like HamiltonOp for the time dependent case from a TD potential: now having both stat and td potentials linked to the qmsystem
- [x] added timing decorators
# Rejected
- [-] only one class for operators (only OperatorTD) since it implements all features of the OperatorConst class
- [-] add spin?