import scipy  as sp 
import numpy as np 

# generate a random dimension m
rng = np.random.default_rng()
m = rng.integers(50, high=100)
p = np.random.randint(10, 40) # always p < m
q = np.random.randint(m-p+1, m-1) # always m1 < q < m 
# next line works
# q = np.random.randint(1, m-p-1) # always q < m1
X = sp.stats.unitary_group.rvs(m) # random unitary matrix
U, D, Vt = sp.linalg.cossin(X, p=p, q=q, separate=False)
assert np.allclose(U@D@Vt, X)