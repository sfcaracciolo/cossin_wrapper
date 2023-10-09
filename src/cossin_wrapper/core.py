import scipy as sp 
import numpy as np 
from typing import Literal, Tuple

def cossin(Q: np.ndarray, shape: Tuple[int, int], ret: Literal['full', 'blocks', 'minimal'] = 'full'):

    p, q = shape
    m, _ = Q.shape
    m1, m2 = m - p, m - q 

    (U_1, U_2), theta, (V_1t, V_2t) = sp.linalg.cossin(Q, p=p, q=q, separate=True)

    w = min([p, q, m1, m2])
    fmin = lambda a, b: min([a, b])-w
    fmax = lambda a, b: max([a-b, 0])

    C = sp.sparse.dia_matrix((np.cos(theta), 0), shape=(w,w), dtype=np.float64) 
    S = sp.sparse.dia_matrix((np.sin(theta), 0), shape=(w,w), dtype=np.float64) 

    D_11 = sp.sparse.block_diag((
        sp.sparse.identity(fmin(p,q), format='dia', dtype=np.float64),
        C,
        sp.sparse.csc_array((fmax(p,q), fmax(q,p)), dtype=np.float64)
    ))

    D_21 = sp.sparse.block_diag((
        sp.sparse.csc_array((fmax(m1,q), fmax(q,m1)), dtype=np.float64),
        S,
        sp.sparse.identity(fmin(m1,q), format='dia', dtype=np.float64)
    ))

    D_12 = sp.sparse.block_diag((
        sp.sparse.csc_array((fmax(p,m2), fmax(m2,p)), dtype=np.float64),
        -S,
        -sp.sparse.identity(fmin(m2,p), format='dia', dtype=np.float64)
    ))

    D_22 = sp.sparse.block_diag((
        sp.sparse.identity(fmin(m1, m2), format='dia', dtype=np.float64),
        C,
        sp.sparse.csc_array((fmax(m1,m2), fmax(m2,m1)), dtype=np.float64)
    ))

    if ret == 'full':
        U = sp.sparse.block_diag((U_1, U_2))
        D = sp.sparse.bmat(((D_11, D_12), (D_21, D_22)))
        Vt = sp.sparse.block_diag((V_1t, V_2t))
        return U, D, Vt

    if ret == 'blocks':
        return (U_1, U_2), (D_11, D_12, D_21, D_22), (V_1t, V_2t)

    if ret == 'minimal':
        return (U_1, U_2), (C, S), (V_1t, V_2t)
    

def _cossin(Q: np.ndarray, shape: Tuple[int, int], ret: Literal['full', 'blocks', 'cs'] = 'full'):
    """This fails when m1 < q due to separate=False issue"""
    p, q = shape
    m, _ = Q.shape
    m1, m2 = m - p, m - q 

    U, D, Vt = sp.linalg.cossin(Q, p=p, q=q, separate=False)

    if ret == 'full':
        return U, D, Vt

    w = min([p, q, m1, m2])
    fmin = lambda a, b: min([a, b])-w
    fmax = lambda a, b: max([a-b, 0])

    D_11, D_12 = D[:p, :q], D[:p, q:]
    D_21, D_22 = D[p:, :q], D[p:, q:]

    if ret == 'cs':
        C = sp.sparse.dia_matrix((D_11[fmin(p,q):fmin(p,q)+w, fmin(p,q):fmin(p,q)+w].diagonal(), 0), shape=(w,w), dtype=np.float64) 
        S = sp.sparse.dia_matrix((D_21[fmax(m1,q):fmax(m1,q)+w, fmax(q,m1):fmax(q,m1)+w].diagonal(), 0), shape=(w,w), dtype=np.float64) 
        return C, S 
    
    U_1, U_2 = U[:p, :p], U[p:, p:]
    V_1t, V_2t = Vt[:q, :q], Vt[q:, q:]

    if ret == 'blocks':
        D_11 = sp.sparse.dia_matrix(D_11, dtype=np.float64)
        D_12 = sp.sparse.dia_matrix(D_12, dtype=np.float64)
        D_21 = sp.sparse.dia_matrix(D_21, dtype=np.float64)
        D_22 = sp.sparse.dia_matrix(D_22, dtype=np.float64)
        return (U_1, U_2), (D_11, D_12, D_21, D_22), (V_1t, V_2t)
