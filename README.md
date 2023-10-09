# CS Decomposition

A wrapper of cossin function of SciPy to return the sparse matrices.

## Usage

```python

from cossin_wrapper import cossin

m = rng.integers(50, high=100)
d = np.random.randint(1, m-1)
Q = sp.stats.unitary_group.rvs(m) # get random unitary matrix

U, D, Vt = cossin(Q, (d, m-d), ret='full')
(U_1, U_2), (D_11, D_12, D_21, D_22), (V_1t, V_2t) = cossin(Q, (d, m-d), ret='blocks')
(U_1, U_2), (C, S), (V_1t, V_2t) = cossin(Q, (d, m-d), ret='minimal')

```