# -*- coding: utf-8 -*-
"""
This module provides functions for the solution of the ODE system for simple
contagion on networks with heterogeneous transmission rates within groups.
"""

import numpy as np
from .utils import *
from numba import jit
from scipy.stats import binom
from scipy.integrate import odeint


def get_state_meta(mmax, nmax, ymax, rate, qm, pyn):
    """Return a tuple that encapsulate all meta information about the system

    :param mmax: maximal membership
    :param nmax: maximal group size >= 2
    :param ymax: maximal rate index
    :param rate: array for the infection rate of length ymax+1
    :param qm: array for the membership distribution of length mmax+1
    :param pyn: array for the joint rate and group size distribution of dimension (ymax+1,nmax+1)

    :return state_meta: tuple of useful arrays describing the system
    """
    m = np.arange(0,mmax+1)
    imat = np.zeros((nmax+1,nmax+1))
    nmat = np.zeros((nmax+1,nmax+1))
    for n in range(2, nmax+1):
        imat[n,0:n+1] = np.arange(n+1)
        nmat[n,0:n+1] = np.ones(n+1)*n
    itens = np.tile(imat, (ymax+1,1,1))
    # print('itens : ', itens)
    ntens = np.tile(nmat, (ymax+1,1,1))
    # print('ntens : ', ntens)
    pn = np.sum(pyn, axis=0)
    pnmat = np.outer(pn,np.ones(nmax+1))
    pntens = np.tile(pnmat, (ymax+1,1,1))
    # print('pntens : ', pntens)
    ratetens = np.repeat(rate,(nmax+1)**2).reshape((ymax+1,nmax+1,nmax+1))
    # print('ratetens : ', ratetens)
    pyntens = np.repeat(pyn,nmax+1).reshape((ymax+1,nmax+1,nmax+1))
    # print('pyntens : ', pyntens)
    return (mmax,nmax,ymax,rate,m,np.array(qm),np.array(pn),np.array(pyn),itens,
            ntens,pntens,ratetens,pyntens)


@jit(nopython=True)
def flatten(Sm,Gyni,state_meta):
    nmax = state_meta[1]
    ymax = state_meta[2]
    return np.concatenate((Sm,Gyni.reshape((ymax+1)*(nmax+1)**2)))


def unflatten(v,state_meta):
    mmax = state_meta[0]
    nmax = state_meta[1]
    ymax = state_meta[2]
    return v[:mmax+1],v[mmax+1:].reshape((ymax+1,nmax+1,nmax+1))


def initialize(state_meta, initial_density=0.5):
    """initialize returns an array representing the state of the
    system at t=0 and state meta information, assuming uniformly distributed
    infected nodes.

    :param state_meta: tuple of arrays encoding information of the structure.
    :param initial_density: float for initial fraction of infected

    :returns (Sm,Gyni): tuple of arrays of shape mmax+1 and
                                  (nmax+1, nmax+1) representing the state of
                                  nodes and groups
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    ymax = state_meta[2]
    qm = state_meta[5]
    pyn = state_meta[7]

    Sm = np.zeros(mmax+1)
    Gyni = np.zeros((ymax+1,nmax+1,nmax+1))
    #initialize nodes
    Sm += (1-initial_density)*qm
    #initialize groups
    for y in range(0,ymax+1):
        for n in range(2, nmax+1):
            pmf = binom.pmf(np.arange(n+1,dtype=int),n,initial_density)
            Gyni[y,n,:n+1] = pmf*pyn[y,n]
    return Sm,Gyni

@jit(nopython=True)
def get_rho(Sm, Gyni, state_meta):
    #unpack
    m = state_meta[4]
    itens = state_meta[8]
    ntens = state_meta[9]
    ratetens = state_meta[11]

    #calculate mean-field quantities
    r = np.sum(ratetens*itens*(ntens-itens)*Gyni)
    r /= np.sum((ntens-itens)*Gyni)
    rho = r*excess_susceptible_membership(m,Sm)

    return rho


def advance(Sm, Gyni, tvar, state_meta):
    """advance integrates the ODE starting from a certain initial state and
    returns the new state.

    :param Sm: array of shape (1,mmax+1) representing the nodes state.
    :param Gyni: array of shape (nmax+1,nmax+1) representing the groups state.
    :param tvar: float for time variation.
    :param state_meta: tuple of arrays encoding information of the structure.

    return (Sm,Gyni): tuple of state arrays later in time
    """
    v = flatten(Sm,Gyni,state_meta)
    t = np.linspace(0,tvar)
    vvec = odeint(vector_field,v,t,args=(state_meta,))
    return unflatten(vvec[-1],state_meta)



@jit(nopython=True)
def vector_field(v, t, state_meta, model='SIS'):
    """vector_field returns the temporal derivative of a flatten state vector

    :param v: array of shape (1,mmax+1+(nmax+1)**2) for the flatten state vector
    :param t: float for time (unused)
    :param state_meta: tuple of arrays encoding information of the structure.

    :returns vec_field: array of shape (1,(nmax+1)**2) for the flatten
                        vector field.
    """
    #unpack
    mmax = state_meta[0]
    nmax = state_meta[1]
    ymax = state_meta[2]
    m = state_meta[4]
    qm = state_meta[5]
    itens = state_meta[8]
    ntens = state_meta[9]
    ratetens = state_meta[11]

    #unflatten
    Sm = v[:mmax+1]
    Gyni = v[mmax+1:].reshape(ymax+1,nmax+1,nmax+1)
    Gyni_field = np.zeros(Gyni.shape) #tensor field
    Sm_field = np.zeros(Sm.shape)

    #calculate mean-field quantities
    r = np.sum(ratetens*itens*(ntens-itens)*Gyni)
    r /= np.sum((ntens-itens)*Gyni)
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
        Gyni_field[:,:,:nmax] += itens[:,:,1:]*Gyni[:,:,1:]
    if model=='SIR':
        Gyni_field[:,:nmax,:nmax] += itens[:,1:,1:]*Gyni[:,1:,1:]
    #contribution from equal
    Gyni_field += (-itens - (ntens - itens)*(ratetens*itens + rho))*Gyni
    #contribution from below
    Gyni_field[:,:,1:] += ((ntens[:,:,:nmax] - itens[:,:,:nmax])
                            *(ratetens[:,:,:nmax]*itens[:,:,:nmax]
                              + rho))*Gyni[:,:,:nmax]
    return np.concatenate((Sm_field,Gyni_field.reshape((ymax+1)*(nmax+1)**2)))

