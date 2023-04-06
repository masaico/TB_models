import numpy as np
import numpy.linalg as la
import gc
import sys
from tqdm import tqdm


def get_Ek(dispersion, kx, ky=None, kz=None):
    # dispersion: func input: KX,KY,KZ (N,) -> output: Ek (N,M,M)
    # k: arr shape: (N,) | (N,N) | (N,N,N), ...
    if ky is None:
        ky = np.zeros(1)
    if kz is None:
        kz = np.zeros(1)
    N = len(kx)*len(ky)*len(kz)
    KX,KY,KZ=map(np.ravel, np.meshgrid(*[kx,ky,kz],indexing="ij"))
    Ek = dispersion(
        KX,KY,KZ
    )
    assert len(Ek.shape) == 3, "dispertion is inadequate."
    Ek = Ek[:,np.newaxis]
    return Ek

def get_local_latt_GreenFunc(Ek, eps, Sigma, N_band=1, return_coherent_GF=False):
    # Ek: arr shape: (N,1,M,M)
    # eps: float | arr shape: (Neps,)
    # Sigma: arr shape: (Neps,M,M) | (M,M)
    eps = np.array(eps).reshape(-1,1,1)*np.eye(N_band)
    coherent_GF = la.inv(eps-Ek-Sigma)  # shape: (N,Neps,M,M)
    del Ek
    gc.collect()
    if return_coherent_GF:
        return coherent_GF
    GF = np.mean(coherent_GF, axis=0)
    del coherent_GF
    gc.collect()
    return GF  # shape: (Neps,M,M)

def get_locater_GreenFunc_inv(GF, Sigma):
    # GF: arr shape: (Neps,M,M)
    # Sigma: arr shape: (Neps,M,M)
    return la.inv(GF) + Sigma  # shape: (Neps,M,M)

def get_impurity_GF(locater_GF_inv, V):
    # locater_GF_inv: arr shape: (Neps,M,M)
    # V: arr shape: (N,M,M)
    return np.mean(la.inv(locater_GF_inv-V[:,np.newaxis]), axis=0)

def get_Self_Energy(locater_GF_inv, impurity_GF):
    # locater_GF_inv: arr shape: (Neps,M,M)
    # impurity_GF: arr shape: (Neps,M,M)
    return locater_GF_inv - la.inv(impurity_GF)

def eval_GF_err(GF, impurity_GF):
    return np.max( np.absolute(GF - impurity_GF) /(np.absolute(GF) + 1e-8) )

def CPA_loop(dispersion, V, eps, kx, ky=None, kz=None, Sigma_init=None, tol=1e-5, maxiter=1000, progressbar=False, return_GF=False):
    N_band = V.shape[-1]
    if Sigma_init is None:
        Sigma = np.array(-1e-4j).reshape(-1,1,1)*np.eye(N_band)
    else:
        Sigma = Sigma_init
    assert len(V.shape) == 3, "V is inadequate."
    Ek = get_Ek(dispersion, kx, ky, kz)
    _range = tqdm(range(maxiter)) if progressbar else range(maxiter)
    for i in _range:
        GF = get_local_latt_GreenFunc(Ek, eps, Sigma, N_band)
        g_loc_inv = get_locater_GreenFunc_inv(GF, Sigma)
        GF_imp = get_impurity_GF(g_loc_inv, V)
        if eval_GF_err(GF, GF_imp) < tol:
            if return_GF:
                return GF, Sigma
            return Sigma
        Sigma = get_Self_Energy(g_loc_inv, GF_imp)

    print("iteration reaches maxiter. err={}".format(eval_GF_err(GF, GF_imp)))
    sys.exit()