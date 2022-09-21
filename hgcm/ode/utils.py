import numpy as np
from numba import jit

@jit(nopython=True)
def infected_fraction(Sm):
    """Return either the prevalence (SIS) or the incidence (SIR)"""
    return 1-np.sum(Sm)

def prevalence_from_incidence(incidence,t):
    dt = t[1]-t[0]
    Svec = 1 - np.array(incidence)
    Ivec = np.zeros(Svec.shape)
    Ivec[0] = incidence[0]
    for j in range(len(t)-1):
        Ivec[j+1] = Ivec[j]*np.exp(-dt) + (Svec[j]-Svec[j+1])
    return Ivec

@jit(nopython=True)
def excess_susceptible_membership(m,Sm):
    """excess_susceptible_membership return the average membership of a
    node, following a random susceptible node within a group

    :param m: array for the membership.
    :param Sm: array for the probability that a node is of membership m and
               susceptible.
    """
    return np.sum(m*(m-1)*Sm)/np.sum(m*Sm)

