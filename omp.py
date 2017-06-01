"""
omp.py

Author: James Ashton Nichols
Start date: April 2017

Some code to test various orthogonal-matching pursuit (OMP) algorithms in 1 dimension. 
Possibly will extend to 2 dimensions at some point.
"""

import math
import numpy as np
import scipy as sp
from scipy.misc import factorial
import collections

from itertools import *
import inspect
import copy
import time

import pdb

def sin_evaluate(x, freq):
    # nb we allow both freq and x to be np arrays
    # returns an array of size len(x) * len(freq)
    return np.sin(math.pi * np.outer(x, freq)) * math.sqrt(2.0) / (math.pi * freq)

def del_evaluate(x, x0):
    # nb we allow both x0 and x to be np arrays
    # returns an array of size len(x) * len(x0)
    normaliser =  1. / np.sqrt((1. - x0) * x0)
    
    # This is now a matrix of size len(x) * len(x0)
    choice = np.less.outer(x, x0) #np.array([x > x0ref for x0ref in x0])

    lower = normaliser * np.outer(x, (1. - x0))
    upper = normaliser * np.outer((1. - x), x0)
    
    if np.isscalar(choice):
        return lower * choice + upper * (not choice)

    return lower * choice + upper * (~choice)
  
def poly_evaluate(x, k):
    # Normaliser - yes the H1_0 norm of x (x - 1) x^k
    return np.outer((x**(k+1) - x**(k+2)), poly_norm(k))

def poly_norm(k):
    return 1.0 / (8.*k*k*k + 32.*k*k + 39.*k + 13.) / ((2.*k+1.)*(2.*k+3.))

def sin_poly_integral(m, k):
    # Integral from 0 to 1 of x^k sin(m pi x)
    # As usual done in the most horribly numpy way possible
    
    # add the new axis as we're making a 3D matrix which we later sum down
    l = np.arange(0,(k[-1]+1)//2)[:,np.newaxis,np.newaxis]
    m = m[:,np.newaxis]
    
    s = factorial(k) / factorial(k-2*l-1)
    s[np.isinf(s)] = 0.0

    d = -1.0 / (m * m * math.pi * math.pi)
    
    full = (-1)** m * s * (-d ** (l+1))
    
    # now add that last little bit
    twid = factorial(k)
    twid[1:-1:2] = 0.0 # Odd ones set to 0

    twid = twid * (-d**((k+1)//2))
    
    return full.sum(axis=0) - twid

def dot_element(lt, lp, lc, rt, rp, rc):
    dot = 0.0
    if lt == 'H1delta':
        c = 1.0 / np.sqrt(lp * (1.0 - lp))
        if rt == 'H1sin':
            dot += (c[:,np.newaxis] * lc[:,np.newaxis] * rc * sin_evaluate(x = lp, freq = rp)).sum()
        elif rt == 'H1delta':
            dot += (c[:,np.newaxis] * lc[:,np.newaxis] * rc * del_evaluate(x = lp, x0 = rp)).sum()
        elif rt == 'H1poly':
            dot +=  (c[:,np.newaxis] * lc[:,np.newaxis] * rc * poly_evaluate(x = lp, freq = rp)).sum()
    elif lt == 'H1sin':
        if rt == 'H1sin':
            dot += (lc[:,np.newaxis] * rc * np.equal.outer(lp, rp)).sum()
        elif rt == 'H1delta':
            c = 1.0 / np.sqrt(rp * (1.0 - rp))
            dot += (c[:, np.newaxis] * lc * rc[:, np.newaxis] * sin_evaluate(x = rp, freq = lp)).sum()
        elif rt == 'H1poly':
            dot += 0.0 
    elif lt == 'H1poly':
        if rt == 'H1sin':
            dot += 0.0
        elif rt == 'H1delta':
            c = 1.0 / np.sqrt(rp * (1.0 - rp))
            dot += (c[:, np.newaxis] * lc * rc[:, np.newaxis] * poly_evaluate(x = rp, freq = lp)).sum()
        elif rt == 'H1poly':
            l = lp[:, np.newaxis]
            k = rp
            dot += poly_norm(l) * poly_norm(k) * lc[:, np.newaxis] * rc * ((l + 1) * (k + 1) / (l + k + 1) \
                   + ((l + 1) * (k + 1) + (l + 2) * (k + 1)) / (l + k + 2) \
                   + (l + 2) * (k + 2) / (l + k + 3))

    return dot

# Define a basis as a collection of elements

# Write the dictionary and Basis class, and basis pair class, in terms of these elements
# Make that Basis class the same as the Basis class in the dyadic FEM code one day, so that this
# Can all become part of the same library

# Define the OMP algorithm from there on...


class Vector(object):
    
    # Ok new paradigm - use numpy to be a bit faster...

    def __init__(self, params=None, coeffs=None, fn_types=None):
        
        self.params = []
        self.coeffs = []
        self.fn_types = []

        if params is not None and fn_types is not None and coeffs is not None:

            if len(params) != len(fn_types) or len(coeffs) != len(fn_types):
                raise Exception('Need as many parameters and coefficients as func types')

            self.fn_types = fn_types
            self.n_types = len(self.fn_types)
            
            for i in range(self.n_types):
                if np.isscalar(params[i]):
                    # Don't sort if singleton
                    self.params.append(np.array([params[i]]))
                    self.coeffs.append(np.array([coeffs[i]]))
                else:
                    s = np.argsort(params[i])
                    self.params.append(np.array(params[i])[s])
                    self.coeffs.append(np.array(coeffs[i])[s])

    def dot(self, other):
        # To keep values *exact* we do the dot product between all the elements of the dictionary
        d = 0.0
        
        for s_p, s_c, s_ft in zip(self.params, self.coeffs, self.fn_types):
            for o_p, o_c, o_ft in zip(other.params, other.coeffs, other.fn_types):
                d += dot_element(s_ft, s_p, s_c, o_ft, o_p, o_c)

        return d
   
    def norm(self):
        return math.sqrt(self.dot(self))

    def evaluate(self, x):
        val = np.zeros(x.shape)
        for fn_i, fn_type in enumerate(self.fn_types):
            if fn_type == 'H1sin':
                val += (self.coeffs[fn_i] * sin_evaluate(x, self.params[fn_i])).sum(axis=-1)
                #for p, c in zip(self.params[fn_i], self.coeffs[fn_i]):
                #    val += c * sin_evaluate(x, p)
                #for p_i in range(len(self.params[fn_i])):
                #    val += self.coeffs[fn_i][p_i] * sin_evaluate(x, self.params[fn_i][p_i])
            if fn_type == 'H1delta':
                val += (self.coeffs[fn_i] * del_evaluate(x, self.params[fn_i])).sum(axis=-1)
                #for p, c in zip(self.params[fn_i], self.coeffs[fn_i]):
                #    val += c * del_evaluate(x, p)
                #for p_i in range(len(self.params[fn_i])):
                #    val += self.coeffs[fn_i][p_i] * del_evaluate(x, self.params[fn_i][p_i])
        return val


    def merge_type(self, p, c, fn_type):

        if fn_type not in self.fn_types:
            self.fn_types.append(fn_type)
            self.params.append(np.array([]))
            self.coeffs.append(np.array([]))
        
        i = self.fn_types.index(fn_type)

        # The strange task of merging our sorted numpy arrays...
        self.params[i] = np.append(self.params[i], p)
        self.coeffs[i] = np.append(self.coeffs[i], c)

        s = np.argsort(self.params[i])

        self.params[i] = self.params[i][s]
        self.coeffs[i] = self.coeffs[i][s]
        
        self.params[i], inv = np.unique(self.params[i], return_inverse=True)
        self.coeffs[i] = np.bincount(inv, self.coeffs[i])
       
    def __add__(self, other):
        result = copy.deepcopy(self)
        for o_fn_i, fn_type in enumerate(other.fn_types):
            result.merge_type(other.params[o_fn_i], other.coeffs[o_fn_i], fn_type)

        return result

    __radd__ = __add__

    def __iadd__(self, other):
        for o_fn_i, fn_type in enumerate(other.fn_types):
            self.merge_type(other.params[o_fn_i], other.coeffs[o_fn_i], fn_type)

        return self 
     
    def __sub__(self, other):
        result = copy.deepcopy(self)
        for o_fn_i, fn_type in enumerate(other.fn_types):
            result.merge_type(other.params[o_fn_i], -other.coeffs[o_fn_i], fn_type)

        return result

    __rsub__ = __sub__

    def __isub__(self, other):
        for o_fn_i, fn_type in enumerate(other.fn_types):
            self.merge_type(other.params[o_fn_i], -other.coeffs[o_fn_i], fn_type)

        return self 

    def __neg__(self):
        result = copy.deepcopy(self)
        for c in result.coeffs:
            c = -c
        return result
 
    def __pos__(self):
        result = copy.deepcopy(self)
        for c in result.coeffs:
            c = +c

    def __mul__(self, other):
        result = copy.deepcopy(self)
        for c in result.coeffs:
            c *= other
        return result

    __rmul__ = __mul__

    def __truediv__(self, other):
        result = copy.deepcopy(self)
        for c in self.coeffs:
            c /= other
        return result

class Basis(object):
    """ A vaguely useful encapsulation of what you'd wanna do with a basis,
        including an orthonormalisation procedure """

    def __init__(self, vecs=None):
        
        if vecs is not None:
            self.vecs = vecs
            self.n = len(vecs)

        self.orthonormal_basis = None
        self.G = None
        self.U = self.S = self.V = None

    def add_vector(self, vec):
        """ Add just one vector, so as to make the new Grammian calculation quick """

        self.vecs.append(vec)
        self.n += 1

        if self.G is not None:
            self.G = np.pad(self.G, ((0,1),(0,1)), 'constant')
            for i in range(self.n):
                self.G[self.n-1, i] = self.G[i, self.n-1] = self.vecs[-1].dot(self.vecs[i])

        self.U = self.V = self.S = None

    def subspace(self, indices):
        """ To be able to do "nested" spaces, the easiest way is to implement
            subspaces such that we can draw from a larger ambient space """
        sub = type(self)(self.vecs[indices])
        if self.G is not None:
            sub.G = self.G[indices, indices]
        return sub

    def subspace_mask(self, mask):
        """ Here we make a subspace by using a boolean mask that needs to be of
            the same size as the number of vectors. Used for the cross validation """
        if mask.shape[0] != len(self.vecs):
            raise Exception('Subspace mask must be the same size as length of vectors')

        sub = type(self)(list(compress(self.vecs, mask)))
        if self.G is not None:
            sub.G = self.G[mask,mask]
        return sub

    def dot(self, u):
        u_d = np.zeros(self.n)
        for i, v in enumerate(self.vecs):
            u_d[i] = v.dot(u)
        return u_d

    def make_grammian(self):
        if self.G is None:
            self.G = np.zeros([self.n,self.n])
            for i in range(self.n):
                for j in range(i+1):
                    self.G[i,j] = self.G[j,i] = self.vecs[i].dot(self.vecs[j])

    def cross_grammian(self, other):
        CG = np.zeros([self.n, other.n])
        
        for i in range(self.n):
            for j in range(other.n):
                CG[i,j] = self.vecs[i].dot(other.vecs[j])
        return CG

    def project(self, u, return_coeffs=False):
        
        # Either we've made the orthonormal basis...
        if self.orthonormal_basis is not None:
            return self.orthonormal_basis.project(u) 
        else:
            if self.G is None:
                self.make_grammian()

            u_n = self.dot(u)
            try:
                if sp.sparse.issparse(self.G):
                    y_n = sp.sparse.linalg.spsolve(self.G, u_n)
                else:
                    y_n = sp.linalg.solve(self.G, u_n, sym_pos=True)
            except np.linalg.LinAlgError as e:
                print('Warning - basis is linearly dependent with {0} vectors, projecting using SVD'.format(self.n))

                if self.U is None:
                    if sp.sparse.issparse(self.G):
                        self.U, self.S, self.V =  sp.sparse.linalg.svds(self.G)
                    else:
                        self.U, self.S, self.V = np.linalg.svd(self.G)
                # This is the projection on the reduced rank basis 
                y_n = self.V.T @ ((self.U.T @ u_n) / self.S)

            # We allow the projection to be of the same type 
            # Also create it from the simple broadcast and sum (which surely should
            # be equivalent to some tensor product thing??)
            #u_p = type(self.vecs[0])((y_n * self.values_flat).sum(axis=2)) 
            
            if return_coeffs:
                return self.reconstruct(y_n), y_n

            return self.reconstruct(y_n)

    def reconstruct(self, c):
        # Build a function from a vector of coefficients
        if len(c) != len(self.vecs):
            raise Exception('Coefficients and vectors must be of same length!')
         
        u_p = Vector()
        for i, c_i in enumerate(c):
            u_p += c_i * self.vecs[i] 
        return u_p

    def matrix_multiply(self, M):
        # Build another basis from a matrix, essentially just calls 
        # reconstruct for each row in M
        if M.shape[0] != M.shape[1] or M.shape[0] != self.n:
            raise Exception('M must be a {0}x{1} square matrix'.format(self.n, self.n))

        vecs = []
        for i in range(M.shape[0]):
            vecs.append(self.reconstruct(M[i,:]))
        
        return Basis(vecs)

    def ortho_matrix_multiply(self, M):
        # Build another basis from an orthonormal matrix, 
        # which means that the basis that comes from it
        # is also orthonormal *if* it was orthonormal to begin with
        if M.shape[0] != M.shape[1] or M.shape[0] != self.n:
            raise Exception('M must be a {0}x{1} square matrix'.format(self.n, self.n))

        vecs = []
        for i in range(M.shape[0]):
            vecs.append(self.reconstruct(M[i,:]))
        
        # In case this is an orthonormal basis
        return type(self)(vecs)

    def orthonormalise(self):

        if self.G is None:
            self.make_grammian()
        
        # We do a cholesky factorisation rather than a Gram Schmidt, as
        # we have a symmetric +ve definite matrix, so this is a cheap and
        # easy way to get an orthonormal basis from our previous basis
        
        if sp.sparse.issparse(self.G):
            L = sp.sparse.cholmod.cholesky(self.G)
        else:
            L = np.linalg.cholesky(self.G)
        L_inv = sp.linalg.lapack.dtrtri(L.T)[0]
         
        ortho_vecs = []
        for i in range(self.n):
            ortho_vecs.append(self.reconstruct(L_inv[:,i]))
                    
        self.orthonormal_basis = OrthonormalBasis(ortho_vecs)

        return self.orthonormal_basis

class OrthonormalBasis(Basis):

    def __init__(self, vecs=None):
        # We quite naively assume that the basis we are given *is* in 
        # fact orthonormal, and don't do any testing...

        super().__init__(vecs=vecs)
        #self.G = np.eye(self.n)
        #self.G = sp.sparse.identity(self.n)

    def project(self, u):
        # Now that the system is orthonormal, we don't need to solve a linear system
        # to make the projection
        return self.reconstruct(self.dot(u))

    def orthonormalise(self):
        return self


class BasisPair(object):
    """ This class automatically sets up the cross grammian, calculates
        beta, and can do the optimal reconstruction and calculated a favourable basis """

    def __init__(self, Wm, Vn, CG=None):

        #if Vn.n > Wm.n:
        #    raise Exception('Error - Wm must be of higher dimensionality than Vn')

        self.Wm = Wm
        self.Vn = Vn
        self.m = Wm.n
        self.n = Vn.n
        
        if CG is not None:
            self.CG = CG
        else:
            self.CG = self.cross_grammian()

        self.U = self.S = self.V = None

    def cross_grammian(self):
        CG = np.zeros([self.m, self.n])
        
        for i in range(self.m):
            for j in range(self.n):
                CG[i,j] = self.Wm.vecs[i].dot(self.Vn.vecs[j])
        return CG
    
    def add_Vn_vector(self, v):
        self.Vn.add_vector(v)
        self.n += 1

        if self.CG is not None:
            self.CG = np.pad(self.CG, ((0,1),(0,0)), 'constant')

            for i in range(self.m):
                self.CG[i, self.n-1] = self.Wm.vecs[i].dot(v)

        self.U = self.V = self.S = None

    def add_Wm_vector(self, w):
        self.Wm.add_vector(w)
        self.m += 1

        if self.CG is not None:
            self.CG = np.pad(self.CG, ((0,0),(0,1)), 'constant')

            for i in range(self.m):
                self.CG[self.m-1, i] = self.Vn.vecs[i].dot(w)

        self.U = self.V = self.S = None

    def beta(self):
        if self.U is None or self.S is None or self.V is None:
            self.calc_svd()

        return self.S[-1]

    def calc_svd(self):
        if self.U is None or self.S is None or self.V is None:
            self.U, self.S, self.V = np.linalg.svd(self.CG)

    def make_favorable_basis(self):
        if isinstance(self, FavorableBasisPair):
            return self
        
        if not isinstance(self.Wm, OrthonormalBasis) or not isinstance(self.Vn, OrthonormalBasis):
            raise Exception('Both Wm and Vn must be orthonormal to calculate the favourable basis!')

        if self.U is None or self.S is None or self.V is None:
            self.calc_svd()

        fb = FavorableBasisPair(self.Wm.ortho_matrix_multiply(self.U.T), 
                                self.Vn.ortho_matrix_multiply(self.V),
                                S=self.S, U=np.eye(self.n), V=np.eye(self.m))
        return fb

    def measure_and_reconstruct(self, u, disp_cond=False):
        """ Just a little helper function. Not sure we really want this here """ 
        u_p_W = self.Wm.dot(u)
        return self.optimal_reconstruction(u_p_W, disp_cond)

    def optimal_reconstruction(self, w, disp_cond=False):
        """ And here it is - the optimal reconstruction """
        try:
            c = scipy.linalg.solve(self.CG.T @ self.CG, self.CG.T @ w, sym_pos=True)
        except np.linalg.LinAlgError as e:
            print('Warning - unstable v* calculation, m={0}, n={1} for Wm and Vn, returning 0 function'.format(self.Wm.n, self.Vn.n))
            c = np.zeros(self.Vn.n)

        v_star = self.Vn.reconstruct(c)

        u_star = v_star + self.Wm.reconstruct(w - self.Wm.dot(v_star))

        # Note that W.project(v_star) = W.reconsrtuct(W.dot(v_star))
        # iff W is orthonormal...
        cond = np.linalg.cond(self.CG.T @ self.CG)
        if disp_cond:
            print('Condition number of G.T * G = {0}'.format(cond))
        
        return u_star, v_star, self.Wm.reconstruct(w), self.Wm.reconstruct(self.Wm.dot(v_star)), cond

class FavorableBasisPair(BasisPair):
    """ This class automatically sets up the cross grammian, calculates
        beta, and can do the optimal reconstruction and calculated a favourable basis """

    def __init__(self, Wm, Vn, S=None, U=None, V=None):
        # We quite naively assume that the basis we are given *is* in 
        # fact orthonormal, and don't do any testing...

        if S is not None:
            # Initialise with the Grammian equal to the singular values
            super().__init__(Wm, Vn, CG=S)
            self.S = S
        else:
            super().__init__(Wm, Vn)
        if U is not None:
            self.U = U
        if V is not None:
            self.V = V

    def make_favorable_basis(self):
        return self

    def optimal_reconstruction(self, w, disp_cond=False):
        """ Optimal reconstruction is much easier with the favorable basis calculated 
            NB we have to assume that w is measured in terms of our basis Wn here... """
        
        w_tail = np.zeros(w.shape)
        w_tail[self.n:] = w[self.n:]
        
        v_star = self.Vn.reconstruct(w[:self.n] / self.S)
        u_star = v_star + self.Wm.reconstruct(w_tail)

        return u_star, v_star, self.Wm.reconstruct(w), self.Wm.reconstruct(self.Wm.dot(v_star))


"""
*****************************************************************************************
All the functions below are for building specific basis systems 
*****************************************************************************************
"""

def make_sin_basis(n):
    V_n = []

    # We want an ordering such that we get (1,1), (1,2), (2,1), (2,2), (2,3), (3,2), (3,1), (1,3), ...
    for i in range(1,n+1):
        v_i = Vector([i], [1.0], ['H1sin'])
        V_n.append(v_i)
            
    return OrthonormalBasis(V_n)


def make_random_delta_basis(n, bounds=None, bound_prop=1.0):

    vecs = []
    
    if bounds is not None:
        bound_points = (bounds[1] - bounds[0]) *  np.random.random(round(n * bound_prop)) + bounds[0]
        
        remain_points = (1.0 - (bounds[1] - bounds[0])) * np.random.random(round(n * (1.0 - bound_prop)))
        # Ooof remain points problem - first left
        if bounds[0] > 0.0:
            remain_l = remain_points[remain_points < bounds[0]]
            remain_r = remain_points[remain_points >= bounds[0]] + bounds[1]
            remain_points = np.append(remain_l, remain_r)
        else:
            remain_points += bounds[1]

        points = np.append(bound_points, remain_points)
    else:
        points = np.random.random(n)
        
    for i in range(n):
        v_i = Vector([points[i]], [1.0], ['H1delta']) 
        vecs.append(v_i)
    
    return Basis(vecs)

"""
*****************************************************************************************
All the functions below are for building bases from greedy algorithms. Several
variants are proposed here.
*****************************************************************************************
"""

def make_unif_dictionary(N):

    points, step = np.linspace(0.0, 1.0, N+1, endpoint=False, retstep=True)
    #points = points + 0.5 * step # Make midpoints... don't want 0.0 or 1.0
    points = points[1:] # Get rid of that first one!

    dic = [Vector([p],[1.0],['H1delta']) for p in points]

    return dic

def make_rand_dictionary(N):

    points = np.random.random(N)

    dic = [Vector([p],[1.0],['H1delta']) for p in points]

    return dic

class GreedyBasisConstructor(object):
    """ Probably should rename this class, but it implements the Collective OMP algorithm for constructing Wm """

    def __init__(self, m, dictionary, Vn, verbose=False, remove=True):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
            
        self.dictionary = copy.copy(dictionary)

        self.m = m
        self.Vn = Vn

        self.verbose = verbose
        self.remove = remove
        self.greedy_basis = None
        self.sel_crit = np.zeros(m)

    def initial_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
    
        norms = np.zeros(len(self.dictionary))
        for i in range(len(self.dictionary)):
            for phi in self.Vn.vecs:
                norms[i] += phi.dot(self.dictionary[i]) ** 2

        n0 = np.argmax(norms)

        return n0, norms[n0]

    def next_step_choice(self, i):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """

        proj_t = 0.0
        dot_t = 0.0
        next_crit = np.zeros(len(self.dictionary))
        # We go through the dictionary and find the max of || f ||^2 - || P_Vn f ||^2
        for phi in self.Vn.vecs:
            phi_perp = phi - self.greedy_basis.project(phi)
            for j in range(len(self.dictionary)):
                next_crit[j] += phi_perp.dot(self.dictionary[j]) ** 2
                #p_V_d[i] = self.greedy_basis.project(self.dictionary[i]).norm()
        
        ni = np.argmax(next_crit)

        if self.verbose:
            print('{0} : \t {1}'.format(i, next_crit[ni]))

        return ni, next_crit[ni]

    def construct_basis(self):
        " The construction method should be generic enough to support all variants of the greedy algorithms """
        
        if self.greedy_basis is None:
            n0, self.sel_crit[0] = self.initial_choice()
            
            self.greedy_basis = Basis([self.dictionary[n0]])
            self.greedy_basis.make_grammian()
 
            if self.remove:
                del self.dictionary[n0]

            if self.verbose:
                print('\n\nGenerating basis from greedy algorithm with dictionary: ')
                print('i \t || P_Vn (w - P_Wm w) ||')

            for i in range(1, self.m):
                
                ni, self.sel_crit[i] = self.next_step_choice(i)
                   
                self.greedy_basis.add_vector(self.dictionary[ni])
 
                if self.remove:
                    del self.dictionary[ni]
                       
            if self.verbose:
                print('\n\nDone!')
        else:
            print('Greedy basis already computed!')
        
        return self.greedy_basis


class WorstCaseOMP(GreedyBasisConstructor):
    """ Now the slightly simpler (to analyse) parallel OMP that looks at Vn vecs individually """

    def __init__(self, m, dictionary, Vn, verbose=False, remove=True):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        super().__init__(m, dictionary, Vn, verbose, remove)
            
        self.dictionary = copy.copy(dictionary)

        self.Vtilde = []

        self.BP = None

    def initial_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        v0 = self.Vn.vecs[0]

        dots = np.zeros(len(self.dictionary))
        for i in range(len(self.dictionary)):
            dots[i] = v0.dot(self.dictionary[i])

        n0 = np.argmax(dots)
      
        self.Vtilde.append(v0)

        return n0, dots[n0]

    def next_step_choice(self, i):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """
        
        next_crit = np.zeros(len(self.dictionary))
        # We go through the dictionary and find the max of || f ||^2 - || P_Vn f ||^2
        BP = BasisPair(self.greedy_basis.orthonormalise(), self.Vn)
        FB = BP.make_favorable_basis()
        
        # This corresponds with vector with the smallest singular value from the SVD
        v = FB.Vn.vecs[-1]

        v_perp = v - self.greedy_basis.project(v)
        for j in range(len(self.dictionary)):
            next_crit[j] = abs(v_perp.dot(self.dictionary[j]))
        
        ni = np.argmax(next_crit)
        self.greedy_basis.add_vector(self.dictionary[ni])

        if self.verbose:
            print('{0} : \t {1}'.format(i, next_crit[ni]))

        return ni, next_crit[ni]

