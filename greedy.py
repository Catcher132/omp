import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import omp

N = 1e6
dictionary = omp.make_unif_dictionary(N)
rand_dictionary = omp.make_rand_dictionary(N)

ns = [5, 10, 20, 40, 100]
m_mult = 10

for n in ns:

    m = m_mult * n
    Vn = omp.make_sin_basis(n)

    gbc = omp.GreedyBasisConstructor(m, dictionary, Vn, verbose=True)
    Wm_omp = gbc.construct_basis()

    # Save the omp points
    omp_x = [vec.params[0][0] for vec in Wm_omp.vecs]
    np.save('omp_x_unif_{0}'.format(n), omp_x)

    Wm_omp = Wm_omp.orthonormalise()

    gbc = omp.GreedyBasisConstructor(m, rand_dictionary, Vn, verbose=True)
    Wm_omp = gbc.construct_basis()

    # Save the omp points
    omp_x = [vec.params[0][0] for vec in Wm_omp.vecs]
    np.save('omp_x_rand_{0}'.format(n), omp_x)

    Wm_omp = Wm_omp.orthonormalise()


