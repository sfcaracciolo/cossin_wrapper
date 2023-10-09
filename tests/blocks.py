import numpy as np 
import scipy  as sp 
from src.cossin_wrapper import cossin

rng = np.random.default_rng()

for _ in range(100):

    m = rng.integers(50, high=100)
    p = np.random.randint(1, m-1)
    q = np.random.randint(1, m-1)
    Q = sp.stats.unitary_group.rvs(m)
    (U_1, U_2), (D_11, D_12, D_21, D_22), (V_1t, V_2t) = cossin(Q, (p, q), ret='blocks')
    U = sp.sparse.bmat(((U_1, None),(None, U_2)))
    D = sp.sparse.bmat(((D_11, D_12),(D_21, D_22)))
    Vt = sp.sparse.bmat(((V_1t, None),(None, V_2t)))
    assert np.allclose((U@D@Vt).todense(), Q)

print('OK')