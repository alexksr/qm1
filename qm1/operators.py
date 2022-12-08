import numpy as np
from scipy import sparse
from qm1.basics import timeit_multiple
from qm1.grid import Grid
from qm1.wavefunction import Wavefunction
from qm1.qmsystem import QMSystem
from typing import Union, Callable
from matplotlib.animation import FuncAnimation, FFMpegWriter



class OperatorConst:
  """
  Generic operator class for (time-)constant operators.
  Features:
   - in matrix representation (operates on the value vector of the function of the 1-dimensional grid )
   - arithmetic operations supported
   - differentiation supported
   - uses sparse matrix representation (`csr` format from `scipy.sparse`)
  """
  def __init__(self, grid:Grid, hermitian:bool=True):
    self.grid = grid
    self.sparse_mat = sparse.lil_matrix((self.grid.num, self.grid.num))

  def copy(self)->"OperatorConst":
    result = OperatorConst(self.grid)
    result.sparse_mat = 0 + self.sparse_mat
    return result

  def __mul__(self, other):
    """ multiply a operator by a constant or element wise with another Oprator: used for for combining operators to more complex operators  """
    result = self.copy()
    if isinstance(other, OperatorConst):
      result.sparse_mat = other.sparse_mat * self.sparse_mat
    elif isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
      result.sparse_mat = other * self.sparse_mat
    elif isinstance(other, OperatorTD) and other.num == 0:
      # can only mul a OperatorTD with a constant Operator: only `TDOp*ConstOp` allowed
      # when other.num==0 other only holds a ConstOperator
      # mul the constant part
      result = self * other.constOP
    else:
      raise NotImplementedError('OperatorConst.__mul__: unknown `other` factor. The `other` might be of type `OperatorTD` with nontrivial time dependence for which multiplication is not implemented.Type of other=', type(other))
    return result
  
  def __neg__(self):
    result = self.copy()
    result.sparse_mat = -self.sparse_mat
    return result

  def __rmul__(self, other):
    """ __mul__ is fully commutative with itself and scalars """
    return self * other

  def __add__(self, other) -> 'OperatorConst':
    """ 
    Add operator and another quantitiy.
    Relies on the __add__ of sparse_mat
    """
    result = self.copy()
    if isinstance(other, OperatorConst):
      result.sparse_mat = self.sparse_mat + other.sparse_mat
    elif isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
      result.sparse_mat = self.sparse_mat + other
    else: 
      # give it a try whether other.__add__ accepts OperatorConst
      result = other + result
    return result

  def __sub__(self, other) -> 'OperatorConst':
    return self + -other

  def __pow__(self, power:int) -> 'OperatorConst':
    """ exponentiate an operator : used for for combining operators to more complex operators  """
    result = self.copy()
    result.sparse_mat = self.sparse_mat**power
    return result

  def __call__(self, wavefunc:Wavefunction) -> Wavefunction:
    """ apply the operator to a function """
    result = Wavefunction(wavefunc.grid)
    result.func = self.sparse_mat *  wavefunc.func
    return result

  def __str__(self):
    """ the __str__ is borrowed from np.ndarray """
    return self.matrix().__str__()

  def is_local(self):
    """ Checks whether the operator is local (no off-diagonal elements)"""
    mat = self.matrix()
    count = np.count_nonzero(mat - np.diag(np.diagonal(mat)))
    return count==0

  def show(self, file:str=None):
    """
    Save a graphical representation of the matrix to file.
    - only shows the real part
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    zmin = np.min(self.matrix()[1:-1, 1:-1])
    zmax = np.max(self.matrix()[1:-1, 1:-1])
    try:
      c = ax.matshow(self.matrix(), cmap="viridis", vmin=zmin, vmax=zmax)
    except:
      print('only real operators can be visualized in this way')
      return
    fig.colorbar(c, ax=ax)
    if file:
      plt.savefig(file)
      plt.close()
    else:
      plt.show()    

  def get_diag(self):
    """ get the matrix diagonal of the operator """
    return self.sparse_mat.diagonal()

  def matrix(self):
    """return a (printable) dense version of the operator"""
    return self.sparse_mat.todense()

  def from_matrix(self, mat):
    """ return an operator from a matrix"""
    self.sparse_mat = sparse.lil_matrix(mat)

  def set_diag(self, diag:list, offset:list):
    """ set the matrix diagonals of the operator """
    for d, o in zip(diag, offset):
      self.sparse_mat.setdiag(values=d, k=o)
    if self.grid.bc=='vanishing':
      self.sparse_mat[-1, :] = 0.
      self.sparse_mat[0, :] = 0.
      
  def local(self, value:Union[Callable[[float], float],np.ndarray]):
    """ operator representing a local operation """
    if isinstance(value, np.ndarray):
      self.sparse_mat.setdiag(values=value, k=0)
    elif callable(value):
      self.sparse_mat.setdiag(values=np.array([value(_x) for _x in list(self.grid.points)]))
    else:
      self.sparse_mat.setdiag(values=value, k=0)
    if self.grid.bc == 'vanishing':
      self.sparse_mat[-1, :] = 0.
      self.sparse_mat[0, :] = 0.

  def first_deriv(self):
    """ set the matrix to the grids derivative operator """
    self.sparse_mat = self.grid.first_deriv()
 
  def second_deriv(self):
    """ set the matrix to the grids derivative operator """
    self.sparse_mat = self.grid.second_deriv()

  def finalize(self):
    """ make the sparse matrix more efficient with the csr format """
    self.sparse_mat = self.sparse_mat.tocsr()

  
  def eigen_system(self, k:int, init_wf:Wavefunction=None, initialguess_eigval_0=None, hermitian:bool=True):
    """
    return the lowest `k` eigenvalues and eigen vectors from the eigensystem of the operator (in space representation)
     - `lowest`  means smallest real part (`which='SR'`)
    """
    from timeit import default_timer as timer

    # prepare initial data
    if init_wf is None:
      init_vec = None
    else: 
      init_vec = init_wf.func

    if initialguess_eigval_0 is None:
      init_val = initialguess_eigval_0
    elif not init_wf is None:
      init_val = init_wf.expectation_value(self)

    # get eigensystem
    eigval, eigvec = sparse.linalg.eigs(self.sparse_mat, k, sigma=init_val, v0=init_vec, which='SR', tol=1e-10)

    print('self.sparse_mat', self.matrix())

    # remove imaginary parts, since eigen-states and -values of hermitian opertators are real
    if hermitian:
      eigval = np.real(eigval)
      eigvec = np.real(eigvec)
    eigvec = eigvec.T

    # sort 
    if hermitian:
      eigvec = np.array([vec for _, vec in sorted(zip(eigval, eigvec), key=lambda pair: pair[0])])
      eigval = sorted(eigval)
    else:
      eigvec = np.array([vec for _, vec in sorted(zip(eigval, eigvec), key=lambda pair: np.abs(pair[0]))])
      eigval = sorted(eigval)


    # put the np.ndarray data back to wavefunction class
    eigstate = []
    for i in range(k):
      es = Wavefunction(self.grid)
      es.from_array(eigvec[i])
      eigstate.append(es)

    return eigval, eigstate


def IdentityOp(grid: Grid) -> OperatorConst:
  """
   return the identity operator for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.local(1.)
  _op.finalize()
  return _op



def ZeroOp(grid: Grid) -> OperatorConst:
  """
   return the zero operator for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.local(0.)
  _op.finalize()
  return _op

def PositionOp(grid:Grid) -> OperatorConst:
  """
   return the position operator (local) for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.local(grid.points)
  _op.finalize()
  return _op

def GradientOp(grid:Grid) -> OperatorConst:
  """
   return the gradient operator for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.first_deriv()
  _op.finalize()
  return _op

def MomentumOp(grid:Grid) -> OperatorConst:
  """
   return the momentum operator for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.first_deriv()
  _op = _op * (-1j)
  _op.finalize()
  return _op

def LaplaceOp(grid:Grid) -> OperatorConst:
  """
   return the laplace operator for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.second_deriv()
  _op.finalize()
  return _op

def StatPotentialOp(qsys:QMSystem) -> OperatorConst:
  """
    Return the potential operator (local) for a given system (instance of `QMSystem`) 
    When vanishing bounadary conditions are set, add an "infinite" potential well at the lower and upper bounds of the grid
    For periodic b.c. the user must specify a periodic potential, otherwise the potential will jump.
  """
  _op = OperatorConst(qsys.grid)
  _op.local(qsys.stat_pot)
  _op.finalize()
  return _op

def KineticOp(qsys:QMSystem) -> OperatorConst:
  """
    Return the kinetic energy operator for a given system (instance of `QMSystem`) 
  """
  _op = LaplaceOp(qsys.grid)
  _op = _op * (-0.5/qsys.mass)
  _op.finalize()
  return _op

def make_efficient(ops:list):
  """
    Return more efficient Operators.
    Change the matrix representation `.tocsr()` from `scipy.sparse` after setting up the operators.
  """
  for _op in ops:
    _op.finalize()
  return
  

class OperatorTD:
  """
  Operator class for time-dependent operators.
  Can only represent operators of the form 
  $$
  O_t = O_0 + \sum_{k=1}^N f_k(x, t) O_k 
  $$ 
  where $O_k$ is a time-constant operator of class `OperatorConst`.
  $f(x,t)$ will be applied to $O_k$ as matrix with $f(x_i, t)$ on the diagonal.
  Features:
   - in matrix representation (operates on the value vector of the function of the 1-dimensional grid )
   - arithmetic operations supported (see Pitfalls)
   - differentiation supported
   - uses sparse matrix representation (`csr` format from `scipy.sparse`)
   Pitfalls:
   Expressions like $x p$ (position operator times momentum operator) are possible, but $p x$ cannot be implemented directly.
   Rewrite such expressions analytically (commutator relations) into the supported form, if possible.
   Additionally a OperatorTD object cannot be multiplied from left with another operator object. The problem arises from the depencence of the function $func(x,t)$ on $x$.
   E.g. $Derivative-Op * f(x,t) \hat{O}$ would not know how to derive the function AND the operator and cast the result in the supported form. 
   It would have to memorize each Operator from the product - this is NotImplemented yet.
  # TODO: hints for easy overflow since, e.g., $x*t*Id - x*t*Id$ is not automatically reduced to Zero
  # TODO: impl make_efficient
  """

  def __init__(self, grid: Grid, func: callable = None, op: OperatorConst = None):
    self.grid = grid
    # constant part
    self.constOP = ZeroOp(self.grid)
    # number of contributions
    self.num = 0
    # list of TD functions
    self.funcs = []
    # list of constant operators
    self.ops = []
    # if there is a func or constop, readily add it:
    if not func is None and not op is None:
      self.num = 1
      self.funcs.append(func)
      self.ops.append(op)
    elif func is None and not op is None:
      self.constOP = op
    elif not func is None and op is None:
      self.num = 1
      self.funcs.append(func)
      self.ops.append(IdentityOp(self.grid))
    return


  def info(self):
    return type(self), ' with num=', self.num, 'TD parts'

  def __call__(self, t: float, wavefunc: Wavefunction) -> Wavefunction:
    """
    Apply the operator at time t to a (wave) function.
    This creates a new matrix each time and thus is of order $n**2$ in the worst case (dense matrix).
    """
    result = Wavefunction(wavefunc.grid)
    result.func = self.constOP(wavefunc)
    for func, op in zip(self.funcs, self.ops):
      _locconstop = OperatorConst(self.grid).local(lambda _x: func(_x, t))
      result.func = result.func + _locconstop(op(wavefunc))
    return result

  def __str__(self):
    _str = str(type(self))+" object with "
    _str += "non-zero constOP" if np.any(self.constOP.matrix()) else "zero constOP"
    _str += " and " +str(self.num)+" time dependend contributions"
    return _str

  def eval(self, t: float) -> OperatorConst:
    """
    Evaluate the operator at a specific time, returning a constant operator.
    """
    result = self.constOP.copy()
    for func, op in zip(self.funcs, self.ops):
      _locconstop = OperatorConst(self.grid)
      _locconstop.local(lambda _x: func(_x, t))
      result = result + _locconstop * op
    return result

  def sparse_mat(self, t: float) -> 'OperatorConst':
    """
    Evaluate the operator at a specific time, returning the sparse matrix of constant operator.
    """
    return self.eval(t).sparse_mat

  def matrix(self, t: float) -> np.ndarray:
    """
    Evaluate the operator at a specific time, returning the dense matrix of constant operator.
    """
    return self.sparse_mat(t).todense()

  def copy(self):
    result = OperatorTD(self.grid)
    result.constOP = self.constOP + 0 
    result.num = 0 + self.num
    result.funcs = [] + self.funcs
    result.ops = [] + self.ops
    return result


  def __init__(self, grid: Grid, func: callable = None, constop: OperatorConst = None):
    self.grid = grid
    # constant part
    self.constOP = ZeroOp(self.grid)
    # number of contributions
    self.num = 0
    # list of TD functions
    self.funcs = []
    # list of constant operators
    self.ops = []
    # if there is a func or constop, readily add it:
    if not func is None and not constop is None:
      self.num = 1
      self.funcs.append(func)
      self.ops.append(constop)
    elif func is None and not constop is None:
      self.constOP = constop
    elif not func is None and constop is None:
      self.num = 1
      self.funcs.append(func)
      self.ops.append(IdentityOp(self.grid))



  def __add__(self, other) -> 'OperatorTD':
    """
    Add operator and another quantitiy.
    """
    result = self.copy()
    # check behaviour for each instance
    if isinstance(other, OperatorConst):
      # add to the constant part of TDOP
      result.constOP = self.constOP + other
    elif isinstance(other, OperatorTD):
      # also add the constant part
      result.constOP = self.constOP + other.constOP
      # add new TD parts from other (which might have multiple terms)
      result.num = self.num + other.num
      result.funcs = self.funcs + other.funcs
      result.ops = self.ops + other.ops
    elif isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
      result.constOP = self.constOP + other * IdentityOp(self.grid)
    else:
      # give it a try: call other.__add__(result)
      result = other + result
    return result

  def __mul__(self, other) -> 'OperatorTD':
    """ multiply a operator by a constant or element wise with another Operator: used for for combining operators to more complex operators
    $O_T * O_c = ( O_0 + \sum_{k=1}^N f_k(t) O_k ) * O_c = O_0O_c + \sum_{k=1}^N f_k(t) O_kO_c $
    $O_T * \tilde{O}_T = ( O_0 + \sum_{k=1}^N f_k(t) O_k ) * ( \tilde{O}_0 + \sum_{k=1}^N f_k(t) \tilde{O}_k ) =  O_0\tilde{O}_0 + \sum_{k,k=1}^{NB,\tilde{N}} f_k(t)\tilde{f}_k(t) O_k\tilde{O}_k  $
    """
    result = self.copy()
    # check behaviour for each instance
    if isinstance(other, OperatorConst):
      # mul the constant part of TDOP with other
      result.constOP = self.constOP * other
      # then mul the ops via listcomp (this is important, dont use a iteration)
      result.ops = [_op * other for _op in result.ops]
    elif isinstance(other, OperatorTD) and other.num == 0:
      # can only mul a OperatorTD with a constant Operator: only `TDOp*ConstOp` allowed
      # when other.num==0 other only holds a ConstOperator
      # mul the constant part
      result.constOP = self.constOP * other.constOP
      # mul all new TD parts in self.ops with other
      result.ops = [_op * other for _op in result.ops]
    elif isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
      # any other scalar: mul ops with other
      result.constOP = self.constOP * other
      result.ops = [_op * other for _op in result.ops]
    else:
      raise NotImplementedError(
          'OperatorTD.__mul__: unknown `other` factor. Hint: Multiplication of two `OperatorTD` types is only supported if the right Op has no TD part, only a Constant. Type of other=', type(other))
    return result

  def __rmul__(self, other):
    if not (isinstance(other, float) or isinstance(other, int) or isinstance(other, complex)):
      raise NotImplementedError
    return self * other


  def __neg__(self) -> 'OperatorTD':
    """ 
    Negate the operator by negating the matrices. Relies on the negation of `OperatorConst`
    """
    result = self.copy()
    result.ops = [-_op for _op in self.ops]
    return result

  def __sub__(self, other) -> 'OperatorTD':
    """ 
    Subs operator and another quantitiy.
    Relies on self.__add__ and other.__neg__
    """
    return self + -other


  def __pow__(self, power: int) -> 'OperatorTD':
    """ exponentiate an operator : used for for combining operators to more complex operators  """
    result = self.copy()
    for _ in range(power-1):
      result = result * self
    return result


  def is_local(self):
    """ Checks whether the operator is local (no off-diagonal elements). """
    local = self.constOP.is_local()
    for _op in self.ops:
      local = local and _op.is_local()
    return local

  def finalize(self):
    """ make the sparse matrices more efficient with the csr format """
    self.constOP.finalize()
    for _op in self.ops:
      _op.finalize()
    return None

  def show(self, tgrid:np.ndarray, file: str = None) -> Union[None, FuncAnimation]:
    """
    Save a graphical representation of the matrix to file.
    """
    import matplotlib.pyplot as plt
    # disable immediate plotting in interactive mode
    plt.ioff()
    # set a writer
    writer = FFMpegWriter(fps=24)
    # plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    # get color range
    zmin = min([np.min(self.matrix(_t)[1:-1, 1:-1]) for _t in tgrid])
    zmax = max([np.max(self.matrix(_t)[1:-1, 1:-1]) for _t in tgrid])
    zmin, zmax = zmin-0.01*(zmax-zmin), zmax+0.01*(zmax-zmin)

    if self.is_local():
      def animate(i):
        ax.clear()
        ax.set_ylim((zmin, zmax))
        line = ax.plot(self.grid.points, self.eval(t=tgrid[i]).get_diag())
        return [line]
    else:
      def animate(i):
        ax.clear()
        matshow = ax.imshow(self.matrix(t=tgrid[i]), vmin=zmin, vmax=zmax)
        return [matshow]

    ani = FuncAnimation(fig=fig, func=animate, frames=len(tgrid), interval=1000./24.)
    if file: ani.save(file, writer=writer)
    # close the figure
    plt.close()
    # enable plotting again in interactive mode
    plt.ion()
    return ani
    


def TDPotentialOp(qsys:QMSystem)->OperatorTD: 
  """
  return a dipol potential operator from a time dependent callable.  
  """
  _op = OperatorTD(qsys.grid, func=qsys.td_pot)
  return _op


def HamiltonOp(qsys: QMSystem) -> Union[OperatorConst, OperatorTD]:
  """
   return the hamilton operator for a given grid system (instance of `QMSystem`)
  """
  # static part
  _op = KineticOp(qsys) + StatPotentialOp(qsys)

  # add td part if present
  try:
    _op += TDPotentialOp(qsys)
  except AttributeError:
    pass


  # make more efficient
  _op.finalize()
  return _op
