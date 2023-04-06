import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from scipy.special import jv

import sys
import time
from typing import Union, Optional, Callable
from dataclasses import dataclass


@dataclass
class Chebyshev:
    H: sp.csr_matrix
    dt: float
    H_scale: Optional[float]=None
    H_center: Optional[float]=None
    info: bool=True
    tol: float=1e-4
    max_k: int=20
    min_k: int=0
    alpha: float=1e-4

    def __post_init__(self):
        def get_Bessel(dt, tol, H_scale):
            J_list = [jv(0,H_scale*dt)]
            for v in range(1,self.max_k+1):
                J_list.append(2*jv(v,H_scale*dt))
                if np.absolute(J_list[-1]) < tol and v >= self.min_k:
                    break
                if v == self.max_k:
                    print('Error: The required accuracy was not reached.')
                    sys.exit()
            if self.info:
                print('expansion order k = {}'.format(v))
            return np.array(J_list), v

        if self.H_scale is None:
            s = time.time()
            Emax, _ = spla.eigsh(self.H, k=1)
            self.H_scale = np.abs(Emax[0])
            if self.info:
                print('maximum eigenvalue = {:.4g}, computation time = {:.4g} [s]'.format(self.H_scale, time.time()-s))
        if self.H_center is None:
            self.H_center = 0.
        self.H_scale *= 1. + self.alpha
        self.H_tild = (self.H - sp.diags([self.H_center], shape=self.H.shape)) /self.H_scale
        self.J_list, self.v = get_Bessel(self.dt, self.tol, self.H_scale)
        self.phi_list = np.empty((self.v+1, self.H.shape[0]), dtype=np.complex128)
        self.coef = np.exp(-1j*self.H_center*self.dt)

    def get_U(self):
        def get_T(H, v):
            T_list = [
                sp.eye(H.shape[0], dtype=np.complex128, format='csr'), -1j*H.copy()
                ]
            for _ in range(2,v+1):
                T_list.append(-2j*H@T_list[-1] + T_list[-2])
            return T_list
        T_list = get_T(self.H_tild, self.v)
        U_list = []
        for k in range(self.v+1):
            U_list.append(self.J_list[k] *T_list[k])
        return np.sum(U_list, axis=0) *self.coef

    def apply_U(self, psi):
        self.phi_list[0] = psi
        self.phi_list[1] = -1j *self.H_tild.dot(psi)
        for i in range(2, self.v+1):
            self.phi_list[i] = -2j*self.H_tild.dot(self.phi_list[i-1]) + self.phi_list[i-2]
        return np.dot(self.J_list, self.phi_list) *self.coef


def gaussian(X, k0, x0, sigma, norm=True):
    sigma_sq = sigma*sigma
    XX = X - x0
    psi0 = np.exp(-0.25*XX**2/sigma_sq + 1j*k0*XX)
    if norm:
        psi0 /= la.norm(psi0)
    return psi0

def plane_wave(X, k0, norm=True):
    psi0 = np.exp(1j*k0*X)
    if norm:
        psi0 /= la.norm(psi0)
    return psi0