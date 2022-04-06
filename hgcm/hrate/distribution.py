# -*- coding: utf-8 -*-
"""
This module provides functions to define rate distributions.
"""

import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve

def get_joint(py_n,pn):
    pyn = py_n*1.
    for n in range(2,len(pn)):
        pyn[:,n] *= pn[n]
    return pyn

def weibull_rate_distribution(mu,nu,rate,ymax,nmax):
    """weibull_rate_distribution returns an array of size (ymax+1,nmax+1)
    for the rate distribution for each group size.

    :param mu: array for the scale parameter of length nmax+1
    :param nu: array for the shape parameter of length nmax+1
    :param rate: array for the infection rate of length ymax+1
    :param ymax: maximal rate index, fixes linear discretization
    :param nmax: maximal group size >= 2
    """
    mumat = np.outer(np.ones(ymax+1),mu)
    numat = np.outer(np.ones(ymax+1),nu)
    ratemat = np.outer(rate,np.ones(nmax+1))
    py_n = (ratemat/mumat)**(1/numat-1)*np.exp(-(ratemat/mumat)**(1/numat))/\
            (mumat*numat)
    #normalize
    py_n /= np.sum(py_n, axis=0)
    return py_n

def weibull_parameter(mean,variance):
    f = lambda x: np.array([x[0]*gamma(1+x[1]),
                            x[0]**2*(gamma(1+2*x[1]) - gamma(1+x[1])**2)]) -\
                            np.array([mean,variance])
    x0 = np.array([1,1])
    x = fsolve(f,x0)
    return x[0],x[1]

def lognormal_rate_distribution(mu,nu,rate,ymax,nmax):
    """lognormal_rate_distribution returns an array of size (ymax+1,nmax+1)
    for the rate distribution for each group size.

    :param mu: array for the scale parameter of length nmax+1
    :param nu: array for the shape parameter of length nmax+1
    :param rate: array for the infection rate of length ymax+1
    :param ymax: maximal rate index, fixes linear discretization
    :param nmax: maximal group size >= 2
    """
    mumat = np.outer(np.ones(ymax+1),mu)
    numat = np.outer(np.ones(ymax+1),nu)
    ratemat = np.outer(rate,np.ones(nmax+1))
    py_n = np.exp(-(np.log(ratemat/mumat) - numat/2)**2/(2*numat))/\
            (ratemat*np.sqrt(2*np.pi*numat))
    #normalize
    py_n /= np.sum(py_n, axis=0)
    return py_n

def frechet_rate_distribution(mu,nu,rate,ymax,nmax):
    """frechet_rate_distribution returns an array of size (ymax+1,nmax+1)
    for the rate distribution for each group size.

    :param mu: array for the scale parameter of length nmax+1
    :param nu: array for the shape parameter of length nmax+1
    :param rate: array for the infection rate of length ymax+1
    :param ymax: maximal rate index, fixes linear discretization
    :param nmax: maximal group size >= 2
    """
    mumat = np.outer(np.ones(ymax+1),mu)
    numat = np.outer(np.ones(ymax+1),nu)
    ratemat = np.outer(rate,np.ones(nmax+1))
    py_n = (ratemat/mumat)**(-1/numat-1)*np.exp(-(ratemat/mumat)**(-1/numat))/\
            (numat*mumat)
    #normalize
    py_n /= np.sum(py_n, axis=0)
    return py_n

