import numpy as np
from scipy.special import gamma
from scipy.special import binom
from scipy.integrate import quad
from numba import jit

@jit(nopython=True)
def get_Gyni(rate,rho,ymax,nmax,pyn):
    """get_cyni returns the stationary cyni (only for the SIS model)

    :param rate: array for the infection rate of length ymax+1
    :param rho: float for mean-field term
    :param ymax: maximal rate index, fixes linear discretization
    :param nmax: maximal group size >= 2
    :param pyn: array for the joint rate and group size distribution of dimension (ymax+1,nmax+1)
    """
    Gyni = np.zeros((ymax+1,nmax+1,nmax+1))
    for y in range(ymax+1):
        for n in range(2,nmax+1):
            Gyni[y,n,0] = 1.
            Gyni[y,n,1] = n*rho
            for i in range(1,n):
                Gyni[y,n,i+1] = ((i + (n-i)*(i*rate[y]+rho))*Gyni[y,n,i]\
                                  -(n-i+1)*(rate[y]*(i-1)+rho)*Gyni[y,n,i-1])/(i+1)
            Gyni[y,n] /= np.sum(Gyni[y,n])
            Gyni[y,n] *= pyn[y,n]
    return Gyni

@jit(nopython=True)
def get_hyni(rate,ymax,nmax):
    """get_hyni returns the stationary hyni (near critical point, only for the SIS model)

    :param rate: array for the infection rate of length ymax+1
    :param ymax: maximal rate index, fixes linear discretization
    :param nmax: maximal group size >= 2
    """
    hyni = np.zeros((ymax+1,nmax+1,nmax+1))
    for y in range(ymax+1):
        for n in range(2,nmax+1):
            hyni[y,n,1] = n
            for i in range(1,n):
                hyni[y,n,i+1] = hyni[y,n,i]*(n-i)*i*rate[y]/(i+1)
            hyni[y,n,0] = -np.sum(hyni[y,n])
    return hyni

@jit(nopython=True)
def get_leading_eigenvector(excess_membership, rate, pn, pyn, ymax, nmax, nb_iter=100, verbose=False,
                            alpha=0.01, model='SIS'):
    """get_leading_eigenvector returns the leading eigenvector of the
    Jacobian matrix linearized at the absorbing state using the power method.

    :param excess_membership: float for <m(m-1)>/<m>
    :param rate: array for the infection rate of length ymax+1
    :param pn: array for the group size distribution of length nmax+1
    :param pyn: array for the joint rate and group size distribution of dimension (ymax+1,nmax+1)
    :param ymax: maximal rate index, fixes linear discretization
    :param nmax: maximal group size >= 2
    """
    nmean = np.sum(pn*np.arange(nmax+1))
    vyni = np.random.random((ymax+1,nmax+1,nmax+1)) #leading eigenvector
    for n in range(len(pn)):
        if pn[n] == 0:
            vyni[:,n,:] = 0 #initialize at zero?
    ev = None #eigenvalue estimate
    #power method begins
    for _ in range(nb_iter):
        #calculate mean-field quantity
        psi = 0
        for n in range(nmax+1):
            for i in range(1,n):
                psi += np.sum(rate*i*(n-i)*vyni[:,n,i])
        psi *= excess_membership/nmean
        #calculate new eigenvector
        new_vyni = np.zeros((ymax+1,nmax+1,nmax+1))
        for n in range(0,nmax+1):
            for i in range(0,n+1):
                new_vyni[:,n,i] -= (i + rate*i*(n-i))*vyni[:,n,i]
                if i > 1:
                    new_vyni[:,n,i] += rate*(i-1)*(n-i+1)*vyni[:,n,i-1]
                if i < n and model == 'SIS':
                    new_vyni[:,n,i] += (i+1)*vyni[:,n,i+1]
                if n < nmax and model == 'SIR':
                    new_vyni[:,n,i] += (i+1)*vyni[:,n+1,i+1]
                if i == 1:
                    new_vyni[:,n,i] += n*psi*pyn[:,n]
                if i == 0:
                    new_vyni[:,n,i] -= n*psi*pyn[:,n]
        #estimate ev
        ev = np.sum(vyni*new_vyni)
        if verbose:
            print(_,ev)
        #renormalize and reassign
        new_vyni = new_vyni/np.sqrt(np.sum(new_vyni*new_vyni))
        vyni = alpha*new_vyni + vyni
        vyni = vyni/np.sqrt(np.sum(vyni*vyni)) #renormalize again?

    return vyni


@jit(nopython=True)
def stationary_effective_rate(rate,rho,pyn,ymax,nmax):
    """stationary_effective_rate returns the stationary effective rate for a
    given rho and joint rate distribution pyn (SIS model only).

    :param rate: array for the infection rate of length ymax+1
    :param rho: float for mean-field term
    :param pyn: array for the joint rate and group size distribution of dimension (ymax+1,nmax+1)
    :param ymax: maximal rate index, fixes linear discretization
    :param nmax: maximal group size >= 2
    """
    Gyni = get_Gyni(rate,rho,ymax,nmax,pyn)
    eff_rate = np.zeros((nmax+1,nmax+1))
    for n in range(2,nmax+1):
        for i in range(0,n+1):
            if np.any(pyn[:,n] > 0):
                numerator = np.sum(rate*Gyni[:,n,i])
                denominator = np.sum(Gyni[:,n,i])
                eff_rate[n,i] = numerator/denominator
    return eff_rate

@jit(nopython=True)
def critical_effective_rate(rate,pyn,ymax,nmax):
    """critical_effective_rate returns the critical effective rate for a given
    joint rate distribution pyn (SIS model only)

    :param rate: array for the infection rate of length ymax+1
    :param pyn: array for the joint rate and group size distribution of dimension (ymax+1,nmax+1)
    :param ymax: maximal rate index, fixes linear discretization
    :param nmax: maximal group size >= 2
    """
    hyni = get_hyni(rate,ymax,nmax)
    eff_rate = np.zeros((nmax+1,nmax+1))
    for n in range(2,nmax+1):
        for i in range(1,n+1):
            if np.any(pyn[:,n] > 0):
                numerator = np.sum(rate*hyni[:,n,i]*pyn[:,n])
                denominator = np.sum(hyni[:,n,i]*pyn[:,n])
                eff_rate[n,i] = numerator/denominator
    return eff_rate



@jit(nopython=True)
def exact_effective_rate(rate,Gyni,nmax):
    """exact_effective_rate returns the exact effective rate for a given cyni.

    :param rate: array for the infection rate of length ymax+1
    :param Gyni: array of shape (nmax+1,nmax+1) representing the groups state.
    :param nmax: maximal group size >= 2
    """
    eff_rate = np.zeros((nmax+1,nmax+1))
    for n in range(0,nmax+1):
        for i in range(0,n+1):
            if np.any(Gyni[:,n,i] > 0):
                numerator = np.sum(rate*Gyni[:,n,i])
                denominator = np.sum(Gyni[:,n,i])
                eff_rate[n,i] = numerator/denominator
    return eff_rate


#transfer the function above where appropriate? or keep here and explain
#need to adapt nlrate for joint dist case
@jit(nopython=True)
def qs_vector_field(v, t, state_meta, rate, pyn, ymax, rho0=10**(-3),
                            it=100,rtol=10**(-6)):
    """qs_vector_field returns the temporal derivative of a flatten state vector
    where we use the quasi-static (or quasi-steady state) approximation (SIS only).

    ***Important : to be used with nlrate module, be careful at import for
    conflicts***

    :param v: array of shape (1,mmax+1+(nmax+1)**2) for the flatten state vector
    :param t: float for time (unused)
    :param state_meta: tuple of arrays encoding information of the structure.
    :param rate: array for the infection rate of length ymax+1
    :param pyn: array for the joint rate and group size distribution of dimension (ymax+1,nmax+1)
    :param ymax: maximal rate index, fixes linear discretization
    :param rho0: initial rho for iteration
    :param it: maximum number of iteration

    :returns vec_field: array of shape (1,(nmax+1)**2) for the flatten
                        vector field.
    """
    mmax = state_meta[0]
    nmax = state_meta[1]
    m = state_meta[2]
    qm = state_meta[3]
    pn = state_meta[4]
    imat = state_meta[5]
    nmat = state_meta[6]
    pnmat = state_meta[7]
    Sm = v[:mmax+1]
    Gni = v[mmax+1:mmax+1+(nmax+1)**2].reshape(nmax+1,nmax+1)

    dGni = np.zeros(Gni.shape) #matrix field
    dSm = np.zeros(Sm.shape)

    #get infection matrix through convergence
    rho1 = rho0
    inf_mat = stationary_effective_rate(rate,rho1,pyn,ymax,nmax)*np.arange(nmax+1)
    for k in range(it):
        r = np.sum(inf_mat[2:,:]*(nmat[2:,:]-imat[2:,:])*Gni[2:,:])
        r /= np.sum((nmat[2:,:]-imat[2:,:])*Gni[2:,:])
        rho = r*np.sum(m*(m-1)*Sm)/np.sum(m*Sm)
        inf_mat = stationary_effective_rate(rate,rho,pyn,ymax,nmax)*np.arange(nmax+1)
        if abs(rho-rho1)/rho1 < rtol:
            break
        rho1 = rho

    #contribution for nodes
    #------------------------
    dSm = qm - Sm - Sm*m*r

    #contribution for cliques
    #------------------------
    #contribution from above
    dGni[2:,:nmax] += imat[2:,1:]*Gni[2:,1:]
    #contribution from equal
    dGni[2:,:] += (-imat[2:,:]
                        -(nmat[2:,:] - imat[2:,:])
                        *(inf_mat[2:,:] + rho))*Gni[2:,:]
    #contribution from below
    dGni[2:,1:nmax+1] += ((nmat[2:,:nmax] - imat[2:,:nmax])
                               *(inf_mat[2:,:nmax] + rho))*Gni[2:,:nmax]

    return np.concatenate((dSm,dGni.reshape((nmax+1)**2)))
