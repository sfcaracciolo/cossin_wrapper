import numpy as np 
import scipy  as sp 
from src.cossin_wrapper import cossin

rng = np.random.default_rng()

for _ in range(100):

    m = rng.integers(50, high=100)
    p = np.random.randint(1, m-1)
    q = np.random.randint(1, m-1)
    Q = sp.stats.unitary_group.rvs(m)
    U, D, Vt = cossin(Q, (p, q), ret='full')
    assert np.allclose((U@D@Vt).todense(), Q)

print('OK')