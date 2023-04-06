import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys
from functools import reduce

# 2022/10 現在、正方格子のみ

# ===============================================================================

# 1次元の tight-binding model
def _D1d(N, t, bc):
    if N==1:
        return sp.csr_matrix((N,N))
    D1d = sp.diags([t,0,t], [-1,0,1], shape=(N,N), format='lil')
    if bc == 'periodic':
        D1d[0,-1] = D1d[-1,0] = t
    else:
        pass
    return D1d.tocsr()

# x -> x+1 のホッピングのみ
def _D1d_p(N, t, bc):
    if N==1:
        return sp.csr_matrix((N,N))
    D1d = sp.diags([0,t], [0,1], shape=(N,N), format='lil')
    if bc == 'periodic':
        D1d[-1,0] = t
    else:
        pass
    return D1d.tocsr()

# x -> x-1 のホッピングのみ
def _D1d_m(N, t, bc):
    if N==1:
        return sp.csr_matrix((N,N))
    D1d = sp.diags([t,0], [-1,0], shape=(N,N), format='lil')
    if bc == 'periodic':
        D1d[0,-1] = t
    else:
        pass
    return D1d.tocsr()

# ===============================================================================

# Orthogonal class
# d次元 tight-binding model
def get_tight_binding_Hamiltonian(shape, t=-1., bc='periodic'):
    # 周期的境界条件と固定端境界条件をサポートしている。
    assert bc in ['periodic', 'fixed'], "bc should be 'periodic', 'fixed'!"
    # shape には list, tuple 以外に int を用いてもよい。(その場合、1次元を表す)
    if not hasattr(shape, "__iter__"):
        shape = (shape,)
    H = sp.csr_matrix( (np.prod(shape),np.prod(shape)) )
    for i in range(len(shape)):
        D = [
            _D1d(N,t,bc) if i==j else sp.eye(N) for j, N in enumerate(shape)
        ]
        H += reduce(sp.kron, D)
    return H

# Symplectic class
# d次元 Ando model
def get_Ando_Hamiltonian(shape, t1, t2, t=-1., bc='periodic'):
    def _Tx_p(T1,T2):
        T = np.array([
            [ T1, T2],
            [-T2, T1]
        ])
        return sp.csr_matrix(T)
    
    def _Tx_m(T1,T2):
        T = np.array([
            [ T1,-T2],
            [ T2, T1]
        ])
        return sp.csr_matrix(T)

    def _Ty_p(T1,T2):
        T = np.array([
            [T1, -1j*T2],
            [-1j*T2, T1]
        ])
        return sp.csr_matrix(T)
    
    def _Ty_m(T1,T2):
        T = np.array([
            [T1, 1j*T2],
            [1j*T2, T1]
        ])
        return sp.csr_matrix(T)

    assert bc in ['periodic', 'fixed'], "bc should be 'periodic', 'fixed'!"
    # shape には list, tuple 以外に int を用いてもよい。(その場合、1次元を表す)
    if not hasattr(shape, "__iter__"):
        shape = (shape,1)
    if len(shape) == 1:
        shape = (shape[0],1)
    if not np.allclose(t1**2 + t2**2, 1., atol=1e-4):
        print('Warning: t1^2 + t2^2 is not unity. value={:.6g}'.format(t1**2 + t2**2))

    Nx, Ny = shape[:2]
    Eyes = [sp.eye(N) for N in shape]
    Dx_p = [_Tx_p(t1,t2)] + Eyes
    Dx_p[1] = _D1d_p(Nx, t, bc)
    Dx_m = [_Tx_m(t1,t2)] + Eyes
    Dx_m[1] = _D1d_m(Nx, t, bc)
    Dy_p = [_Ty_p(t1,t2)] + Eyes
    Dy_p[2] = _D1d_p(Ny, t, bc)
    Dy_m = [_Ty_m(t1,t2)] + Eyes
    Dy_m[2] = _D1d_m(Ny, t, bc)
    H = np.sum([
        reduce(sp.kron, D) for D in [Dx_p, Dx_m, Dy_p, Dy_m]
    ])
    for i in range(2,len(shape)):
        D = [
            _D1d(N,t,bc) if i==j else sp.eye(N) for j, N in enumerate(shape)
        ]
        D = [sp.eye(2)] + D
        H += reduce(sp.kron, D)
    return H

# Unitary class
# d次元 Hofstadter model
# 磁場は常にz軸方向、ゲージ不変にはしていない。
def get_Hofstadter_Hamiltonian(shape, phi, x=None, t=-1., bc='periodic'):
    assert bc in ['periodic', 'fixed'], "bc should be 'periodic', 'fixed'!"
    # shape には list, tuple 以外に int を用いてもよい。(その場合、1次元を表す)
    if not hasattr(shape, "__iter__"):
        print('Warning: Normal tight-binding Hamiltonian is returned.')
        shape = (shape,1)
    if len(shape) == 1:
        print('Warning: Normal tight-binding Hamiltonian is returned.')
        shape = (shape[0],1)
    assert len(shape) in [2,3], "dimension must be 2 or 3!"
    # phi は -1 ~ 1 の実数をとる。
    assert np.abs(phi) <= 1., 'phi must be between [-1,1]!'
    # サイトの x座標を指定しなければ、自動的に格子定数 1 となる。
    if x is None:
        x = np.arange(shape[0])
    else:
        x = np.array(x).ravel()
        assert len(x) == shape[0], 'size of x must be {}!'.format(shape[0])

    H = sp.csr_matrix( (np.prod(shape),np.prod(shape)) )
    for i in range(len(shape)):
        D = [
            _D1d(N,t,bc) if i==j else sp.eye(N) for j, N in enumerate(shape)
        ]
        if i == 1:
            peierls = np.exp( -2.j*np.pi*phi*x )
            D[0] = sp.diags(peierls)
            D[1] = _D1d_p(shape[1],t,bc)
            Hy = reduce(sp.kron, D)
            H += Hy + Hy.getH()
        else:
            H += reduce(sp.kron, D)
    return H

# オンサイトポテンシャル
def onsite_potential(H, V):
    assert len(V.ravel()) == H.shape[0], 'size of V must be {}!'.format(H.shape[0])
    return H + sp.diags(V.ravel())