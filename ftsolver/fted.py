'''
exact diagonalization solver with grand canonical statistics.
Chong Sun 08/07/17
taking temperature (T) and chemical potential (mu) as input
Chong Sun 01/16/18
'''

import numpy as np
from numpy import linalg as nl
from functools import reduce
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring
from pyscf.fci import direct_uhf
from scipy.optimize import minimize
import datetime
import scipy
import sys
import os

def rdm12s_fted(h1e,g2e,norb,nelec,beta,mu=0.0,symm='UHF',bmax=1e3, \
                dcompl=False,**kwargs):

    '''
    Return the expectation values of energy, RDM1 and RDM2 at temperature T.
    '''

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver
    

    Z = 0.
    E = 0.
    RDM1=np.zeros((2, norb, norb), dtype=np.complex128)
    RDM2=np.zeros((3, norb, norb, norb, norb), dtype=np.complex128)

    # non-interacting case
    if np.linalg.norm(g2e[1]) == 0:
        RDM1, E = FD(h1e,norb,nelec,beta,mu,symm)
        return RDM1, RDM2, E

    if symm != 'UHF' and isinstance(h1e, tuple):
        h1e = h1e[0]
        g2e = g2e[1]

    if beta>bmax:
        e, v = fcisolver.kernel(h1e, g2e, norb, nelec)
        RDM1, RDM2 = fcisolver.make_rdm12s(v,norb,nelec)
        return np.asarray(RDM1), np.asarray(RDM2), e

    # check for overflow
    e0, _ = fcisolver.kernel(h1e,g2e,norb,norb)
    exp_max = (-e0+mu*norb)*beta
    if(exp_max > 700):
        exp_shift = exp_max - 500
    else:
        exp_shift = 0

    # Calculating E, RDM1, Z
    N = 0
    for na in range(0,norb+1):
        for nb in range(0,norb+1):
            ne = na + nb
            ew, ev = diagH(h1e,g2e,norb,(na,nb),fcisolver)
            exp_ = (-ew+mu*(na+nb))*beta
            exp_ -= exp_shift
            ndim = len(ew) 
            Z += np.sum(np.exp(exp_))
            E += np.sum(np.exp(exp_)*ew)
            N += ne * np.sum(np.exp(exp_))
         
            for i in range(ndim):
                dm1, dm2 = fcisolver.make_rdm12s(ev[:,i].copy(),norb,(na,nb))
                dm1 = np.asarray(dm1,dtype=np.complex128)
                dm2 = np.asarray(dm2,dtype=np.complex128)
                RDM1 += dm1*np.exp(exp_[i])
                RDM2 += dm2*np.exp(exp_[i])

    E    /= Z
    N    /= Z
    RDM1 /= Z
    RDM2 /= Z

    #print "%.6f        %.6f"%(1./T, N.real)
    #print "The number of electrons in embedding space: ", N.real

    if not dcompl:
        E = E.real
        RDM1 = RDM1.real
        RDM2 = RDM2.real
    # RDM2 order: aaaa, aabb, bbbb
    return RDM1, RDM2, E


######################################################################
#def solve_spectrum(h1e,h2e,norb,fcisolver):
#    EW = np.empty((norb+1,norb+1), dtype=object)
#    EV = np.empty((norb+1,norb+1), dtype=object)
#    for na in range(0, norb+1):
#        for nb in range(0, norb+1):
#            ew, ev = diagH(h1e,h2e,norb,(na,nb),fcisolver)
#            EW[na, nb] = ew
#            EV[na, nb] = ev
#    return EW, EV

######################################################################
def diagH(h1e,g2e,norb,nelec,fcisolver):
    '''
        exactly diagonalize the hamiltonian.
    '''
    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ndim = na*nb

    eyemat = np.eye(ndim)
    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    Hmat = []
    for i in range(ndim):
        hc = hop(eyemat[i])
        Hmat.append(hc)


    Hmat = np.asarray(Hmat).T
    ew, ev = nl.eigh(Hmat)
    return ew, ev

######################################################################
def FD(h1e,norb,nelec,beta,mu,symm='RHF'):
    #ew, ev = np.linalg.eigh(h1e)
    htot = np.zeros((2*norb, 2*norb))
    htot[:norb,:norb] = h1e[0]
    htot[norb:,norb:] = h1e[1]
            
    ew, ev = np.linalg.eigh(htot)
    def fermi(mu):
        return 1./(1.+np.exp((ew-mu)*beta))
    if beta > 1e3:
        beta = np.inf
        eocc = np.ones(2*norb)
        eocc[nelec:]*=0.
    else:
        eocc = fermi(mu)

    dm1 = np.asarray(np.dot(ev, np.dot(np.diag(eocc), ev.T.conj())), dtype=np.complex128)
    e = np.sum(ew*eocc)
    DM1 = (dm1[:norb, :norb], dm1[norb:, norb:])

    return DM1, e

######################################################################
if __name__ == '__main__':

    import sys
    norb = 4
    nimp = 2
    nelec = 4
    h1e = np.zeros((norb,norb))
    g2e = np.zeros((norb,norb,norb,norb))
    #T = 0.02
    u = float(sys.argv[1])
    mu= u/2
    #mu= 0.0
    for i in range(norb):
        h1e[i,(i+1)%norb] = -1.
        h1e[i,(i-1)%norb] = -1.
    #h1e[0,-1] = 0.
    #h1e[-1,0] = 0.

    #for i in range(norb):
    #    h1e[i,i] += -u/2.

    for i in range(norb):
        g2e[i,i,i,i] = u

    T = 0.01
    dm1,_,e1 = rdm12s_fted((h1e,h1e),(g2e*0, g2e, g2e*0),norb,nelec,T,mu, symm='UHF')
    print e1/norb#+u/2.
