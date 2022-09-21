# -*- coding: utf-8 -*-
"""
This module provides functions for the solution of the ODE system for general
contagion on networks with groups.
"""

import numpy as np
from .utils import *
from numba import jit
from scipy.stats import binom
from scipy.integrate import odeint

def get_state_meta(mmax, nmax, qm, pn):
    """Return a tuple that encapsulate all meta information about the structure

    :param mmax: maximal membership
    :param nmax: maximal group size >= 2
    :param qm: array for the membership distribution of length mmax+1
    :param pn: array for the group size distribution of length nmax+1

    :return state_meta: tuple of useful arrays describing the structure
    """
    m = np.arange(0,mmax+1)
    imat = np.zeros((nmax+1,nmax+1))
    nmat = np.zeros((nmax+1,nmax+1))
    for n in range(0, nmax+1):
        imat[n,0:n+1] = np.arange(n+1)
        nmat[n,0:n+1] = np.ones(n+1)*n
    pnmat = np.outer(pn,np.ones(nmax+1))
    return (mmax,nmax,m,np.array(qm),np.array(pn),imat,nmat,pnmat)


@jit(nopython=True)
def flatten(Sm,Gni,state_meta):
    nmax = state_meta[1]
    return np.concatenate((Sm,Gni.reshape((nmax+1)**2)))


def unflatten(v,state_meta):
    mmax = state_meta[0]
    nmax = state_meta[1]
    return v[:mmax+1],v[mmax+1:].reshape((nmax+1,nmax+1))


def initialize(state_meta, initial_density=0.5):
    """initialize returns an array representing the state of the
    system at t=0 and state meta information, assuming uniformly distributed
    infected nodes.

    :param state_meta: tuple of arrays encoding information of the structure.
    :param initial_density: float for initial fraction of infected

    :returns (Sm,Gni): tuple of arrays of shape mmax+1 and
                                  (nmax+1, nmax+1) representing the state of
                                  nodes and groups, and the state meta data
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    qm = state_meta[3]
    pn = state_meta[4]

    Sm = np.zeros(mmax+1)
    Gni = np.zeros((nmax+1,nmax+1))
    #initialize nodes
    Sm += qm*(1-initial_density)
    #initialize groups
    for n in range(2, nmax+1):
        pmf = binom.pmf(np.arange(n+1,dtype=int),n,initial_density)
        Gni[n][:n+1] = pmf*pn[n]
    return Sm,Gni



def advance(Sm, Gni, tvar, inf_mat, state_meta):
    """advance integrates the ODE starting from a certain initial state and
    returns the new state.

    :param Sm: array of shape (1,mmax+1) representing the nodes state.
    :param Gni: array of shape (nmax+1,nmax+1) representing the groups state.
    :param tvar: float for time variation.
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.
    :param corr: bool to determine if there are correlations.

    return (Sm,Gni): tuple of state arrays later in time
    """
    v = flatten(Sm,Gni,state_meta)
    t = np.linspace(0,tvar)
    vvec = odeint(vector_field,v,t,args=(inf_mat,state_meta))
    return unflatten(vvec[-1],state_meta)


@jit(nopython=True)
def vector_field(v, t, inf_mat, state_meta, model='SIS'):
    """vector_field returns the temporal derivative of a flatten state vector

    :param v: array of shape (1,mmax+1+(nmax+1)**2) for the flatten state vector
    :param t: float for time (unused)
    :param inf_mat: array of shape (nmax+1,nmax+1) representing the infection rate
    :param state_meta: tuple of arrays encoding information of the structure.

    :returns vec_field: array of shape (1,(nmax+1)**2) for the flatten
                        vector field.
    """
    #unpack
    mmax = state_meta[0]
    nmax = state_meta[1]
    m = state_meta[2]
    qm = state_meta[3]
    pn = state_meta[4]
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]

    #unflatten
    Sm = v[:mmax+1]
    Gni = v[mmax+1:].reshape(nmax+1,nmax+1)
    Gni_field = np.zeros(Gni.shape) #matrix field
    Sm_field = np.zeros(Sm.shape)

    #calculate mean-field quantities
    r = np.sum(inf_mat[:,:]*(nmat[:,:]-imat[:,:])*Gni[:,:])
    r /= np.sum((nmat[:,:]-imat[:,:])*Gni[:,:])
    rho = r*excess_susceptible_membership(m,Sm)

    #contribution for nodes
    #------------------------
    if model=='SIS':
        Sm_field = qm - Sm - Sm*m*r
    if model=='SIR':
        Sm_field = -Sm*m*r

    #contribution for groups
    #------------------------
    #contribution from above
    if model=='SIS':
        Gni_field[:,:nmax] += imat[:,1:]*Gni[:,1:]
    if model=='SIR':
        Gni_field[:nmax,:nmax] += imat[1:,1:]*Gni[1:,1:]
    #contribution from equal
    Gni_field[:,:] += (-imat[:,:]
                        -(nmat[:,:] - imat[:,:])
                        *(inf_mat[:,:] + rho))*Gni[:,:]
    #contribution from below
    Gni_field[:,1:nmax+1] += ((nmat[:,:nmax] - imat[:,:nmax])
                               *(inf_mat[:,:nmax] + rho))*Gni[:,:nmax]
    return np.concatenate((Sm_field,Gni_field.reshape((nmax+1)**2)))

