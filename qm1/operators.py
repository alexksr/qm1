from calendar import c
import numpy as np
from scipy import sparse
from qm1.grid import Grid
from qm1.wavefunction import Wavefunction
from qm1.qmsystem import QMSystem





class OperatorConst:
  """
  Generic operator class for (time-)constant operators.
  Features:
   - in matrix representation (operates on the value vector of the function of the 1-dimensional grid )
   - arithmetic operations supported
   - differentiation supported
   - uses sparse matrix representation (`csr` format from `scipy.sparse`)
  """
  def __init__(self, grid:Grid):
    self.grid = grid
    self.sparse_mat = sparse.lil_matrix((self.grid.num, self.grid.num))

  def __mul__(self, other):
    """ multiply a operator by a constant or element wise with another Oprator: used for for combining operators to more complex operators  """
    result = OperatorConst(self.grid)
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
      raise NotImplementedError('OperatorConst.__mul__: unknown `other` factor. The `other` might be of type `OperatorTD` with nontrivial time dependence for which multiplication is not implemented.')
    return result
  
  def __neg__(self):
    self.sparse_mat = -self.sparse_mat

  def __rmul__(self, other):
    """ __mul__ is fully commutative with itself and scalars """
    return self * other

  def __add__(self, other) -> 'OperatorConst':
    """ 
    Add operator and another quantitiy.
    Relies on the __add__ of sparse_mat
    """
    result = OperatorConst(self.grid)
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
    result = OperatorConst(self.grid)
    result.sparse_mat = self.sparse_mat**power
    return result

  def __call__(self, wavefunc:Wavefunction) -> Wavefunction:
    """ apply the operator to a function """
    result = Wavefunction(wavefunc.grid)
    result.func = self.sparse_mat *  wavefunc.func
    return result

  def __str__(self):
    """write 4x4 blocks of the (upper|lower)x(left|right) blocks of the matrix"""
    header = str(type(self))+" object with values ...\n"
    k=4
    ul_block = [ _str.replace('[[', '[').replace(']]',']').replace(' [', '[') for _str in self.sparse_mat.todense()[:k, :k].__str__().splitlines()]
    ur_block = [ _str.replace('[[', '[').replace(']]',']').replace(' [', '[') for _str in self.sparse_mat.todense()[:k, -k:].__str__().splitlines()]
    ll_block = [ _str.replace('[[', '[').replace(']]',']').replace(' [', '[') for _str in self.sparse_mat.todense()[-k:, :k].__str__().splitlines()]
    lr_block = [ _str.replace('[[', '[').replace(']]',']').replace(' [', '[') for _str in self.sparse_mat.todense()[-k:, -k:].__str__().splitlines()]
    upper, lower = '', ''
    for _i in range(k):
      upper += ul_block[_i]+' ... '+ur_block[_i]+'\n'
      lower += ll_block[_i]+' ... '+lr_block[_i]+'\n'
    return header+upper+' ... \n'+lower

  def show(self, file):
    """
    Save a graphical representation of the matrix to file.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,10))
    c =ax.matshow(self.matrix(), cmap="viridis")
    fig.colorbar(c, ax=ax)
    plt.savefig(file, bbox_inches='tight')
    plt.close()


  def matrix(self):
    """return a (printable) dense version of the operator"""
    return self.sparse_mat.todense()

  def set_diag(self, diag:list, offset:list):
    """ set the matrix diagonals of the operator """
    for d, o in zip(diag, offset):
      self.sparse_mat.setdiag(values=d, k=o)
    if self.grid.bc=='vanishing':
      self.sparse_mat[self.grid.num-1, :] = 0.
      self.sparse_mat[0, :] = 0.

  def set_bounds(self, lb: float, ub: float):
    """ set the value of the operator at lower and upper bounds """
    self.sparse_mat[ 0,  0] = lb
    self.sparse_mat[-1, -1] = ub
      
  def local(self, value):
    """ operator representing a local operation """
    if isinstance(value, np.ndarray):
      self.sparse_mat.setdiag(values=value, k=0)
    elif callable(value):
      self.sparse_mat.setdiag(values=np.array([value(_x) for _x in list(self.grid.points)]))
    else:
      self.sparse_mat.setdiag(values=value, k=0)

  def first_deriv(self):
    """ set the matrix to the grids derivative operator """
    self.sparse_mat = self.grid.first_deriv()
 
  def second_deriv(self):
    """ set the matrix to the grids derivative operator """
    self.sparse_mat = self.grid.second_deriv()

  def finalize(self):
    """ make the sparse matrix more efficient with the csr format """
    self.sparse_mat = self.sparse_mat.tocsr()

  def eigen_system(self, k:int, init_wf:Wavefunction=None, initialguess_eigval_0=None):
    """
    return the lowest `k` eigenvalues and eigen vectors from the eigensystem of the operator (in space representation)
     - `lowest`  means smallest real part (`which='SR'`)
    """

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

    # remove imaginary parts, since eigen-states and -values of hermitian opertators are real
    eigval = np.real(eigval)
    eigvec = np.real(eigvec).T

    # sort 
    eigvec = np.array([vec for _, vec in sorted(zip(eigval, eigvec), key=lambda pair: pair[0])])
    eigval = sorted(eigval)

    # put the np.ndarray data back to wavefunction class
    eigstate = []
    for i in range(k):
      es = Wavefunction(self.grid)
      es.set_via_array(eigvec[i])
      eigstate.append(es)

    return eigval, eigstate


def IdentityOp(grid: Grid):
  """
   return the identity operator  for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.local(1.)
  # make_efficient([_op])
  return _op



def ZeroOp(grid: Grid):
  """
   return the zero operator for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.local(0.)
  return _op

def PositionOp(grid:Grid):
  """
   return the position operator (local) for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.local(grid.points)
  return _op

def GradientOp(grid:Grid):
  """
   return the gradient operator for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.first_deriv()
  return _op

def MomentumOp(grid:Grid):
  """
   return the momentum operator for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.first_deriv()
  _op = _op * (-1j)
  return _op

def LaplaceOp(grid:Grid):
  """
   return the laplace operator for a given system (instance of `QMSystem`) 
  """
  _op = OperatorConst(grid)
  _op.second_deriv()
  return _op

def PotentialOp(qsys:QMSystem):
  """
    Return the potential operator (local) for a given system (instance of `QMSystem`) 
    When vanishing bounadary conditions are set, add an "infinite" potential well at the lower and upper bounds of the grid
    For periodic b.c. the user must specify a periodic potential, otherwise the potential will jump.
  """
  _op = OperatorConst(qsys.grid)
  _op.local(qsys.pot)
  if qsys.grid.bc == 'vanishing':
    _op.set_bounds(lb=1e+9, ub=1e+9)
  return _op

def KineticOp(qsys:QMSystem):
  """
    Return the kinetic energy operator for a given system (instance of `QMSystem`) 
  """
  _op = LaplaceOp(qsys.grid)
  _op = _op * (-0.5/qsys.mass)
  return _op


def HamiltonOp(qsys:QMSystem):
  """
   return the hamilton operator for a given grid system (instance of `QMSystem`)
  """
  return KineticOp(qsys) + PotentialOp(qsys)

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
    result = self.constOP
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

  def __add__(self, other) -> 'OperatorTD':
    """
    Add operator and another quantitiy.
    """
    result = OperatorTD(self.grid)
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
      # any other scalar: add other*unitymatrix
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
    result = OperatorTD(self.grid)
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
    result = self
    self.ops = [-_op for _op in self.ops]
    return result

  def __sub__(self, other) -> 'OperatorTD':
    """ 
    Subs operator and another quantitiy.
    Relies on self.__add__ and other.__neg__
    """
    return self + -other


  def __pow__(self, power: int) -> 'OperatorTD':
    """ exponentiate an operator : used for for combining operators to more complex operators  """
    result = self
    for _ in range(power-1):
      result = result * self
    return result

  def show(self, tgrid, file):
    """
    Save a graphical representation of the matrix to file.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots(figsize=(10, 10))
    # get color range
    zmin = min([np.min(self.matrix(t=tgrid[_i])) for _i in range(len(tgrid))])
    zmax = max([np.max(self.matrix(t=tgrid[_i])) for _i in range(len(tgrid))])
    def animate(i):
        ax.clear()
        matshow = ax.imshow(self.matrix(t=tgrid[i]), vmin=zmin, vmax=zmax)
        return [matshow]
    ani = FuncAnimation(fig=fig, func=animate, frames=len(tgrid), interval=1000./24.)
    ani.save(file, writer='imagemagick', fps=24)
    plt.close()
    
