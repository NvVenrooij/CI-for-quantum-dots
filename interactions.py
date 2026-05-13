import numpy as np
from scipy.spatial.distance import cdist
from numba import jit, prange


@jit(nopython=True, parallel=True)
def compute_dr(x1, x2, y1, y2, z1, z2):
    lx = len(x1)
    dx = np.zeros((lx,lx))
    for i in range(lx):
        for j in range(lx):
            dx[i,j] = x1[i] - x2[j]

    ly = len(y1)
    dy = np.zeros((ly,ly))
    for i in range(ly):
        for j in range(ly):
            dy[i,j] = y1[i] - y2[j]
    
    lz = len(z1)
    dz = np.zeros((lz,lz))
    for i in range(lz):
        for j in range(lz):
            dz[i,j]= z1[i] - z2[j]

    return dx, dy, dz

#@profile
def compute_distance_matrices(xyz1, xyz2):
    """
    Compute distance matrices and inverse distance matrices for a given set of coordinates.

    Parameters:
    - xyz: Array of coordinates (shape: (N, 3)).

    Returns:
    - dist: Distance matrix.
    - invdist: Inverse distance matrix.

    """
    # Define value for vector Delta R between every coordinate
    dr_vec = np.array(compute_dr(np.array(xyz1[:,0]), np.array(xyz2[:,0]) ,
                                 np.array(xyz1[:,1]), np.array(xyz2[:,1]) ,
                                 np.array(xyz1[:,2]), np.array(xyz2[:,2])))


    # Compute the pairwise distances between coordinates (euclidian distance of Delta R)
    dist = cdist(xyz1, xyz2, 'euclidean')

    # Save the locations of original zeros in dr_vec
    zero_locations = np.where(dist == 0)

    # Add a small value to prevent division by zero
    epsilon = 1e-10
    dist[zero_locations] += epsilon

    # Compute the inverse distances
    invdist = dist ** -1

    # Replace 1/epsilon values back to zero in invdist at the original zero locations
    invdist[zero_locations] = 0

    return dr_vec, invdist
 
#@profile
def OL(WF1, WF2):
    """
    Calculate the overlap between two wavefunctions.

    Parameters:
    - WF1: Wavefunction 1.
    - WF2: Wavefunction 2.

    Returns:
    Array of overlap values.
    """
    WF1_dat = np.conj(WF1.data)
    WF2_dat = WF2.data
    return np.sum(WF1_dat * WF2_dat, axis=1, dtype=np.clongdouble)  

#@profile
def P(WF1, WF2):
    """
    Calculate the momentum matrix elements.

    Parameters:
    - WF1: Wavefunction 1.
    - WF2: Wavefunction 2.
    - d_xyz: Array of derivative coordinates (shape: (3, N, 3)).

    Returns:
    Array of momentum matrix elements.

    """

    WF1_dat = np.conj(WF1.data)
    WF2_dat = WF2.data
    d_xyz = WF1.dipole_matrices

    MMEs = np.zeros((WF1_dat.shape[0],3),dtype=np.clongdouble)
    for i in range(3):
        MMEs[:,i] = np.einsum('ki,ij,kj->k', WF1_dat, d_xyz[i], WF2_dat, dtype=np.clongdouble)

    return MMEs

#@profile
def sigma_expectation(WF1, WF2):
    WF1_dat = np.conj(WF1.data[:,:2])
    WF2_dat = WF2.data[:,:2]
    sigma_xyz = WF1.pauli_matrices
    
    WF_sigma_WF = np.zeros((WF1_dat.shape[0],3),dtype=np.clongdouble)
    for i in range(3):
        WF_sigma_WF[:,i] = np.einsum('ki,ij,kj->k', WF1_dat, sigma_xyz[i], WF2_dat, dtype=np.clongdouble)

    return WF_sigma_WF

#@profile
def J_expectation(WF1, WF2):
    WF1_dat = np.conj(WF1.data[:,2:6])
    WF2_dat = WF2.data[:,2:6]
    J_xyz = WF1.J_matrices

    WF_J_WF = np.zeros((WF1_dat.shape[0],3),dtype=np.clongdouble)
    for i in range(3):
        WF_J_WF[:,i] = np.einsum('ki,ij,kj->k', WF1_dat, J_xyz[i], WF2_dat, dtype=np.clongdouble)
    
    return WF_J_WF


#@jit(nopython=True, parallel=True)
#@profile

def J_LR(OLr1, OLr2, invdist):
    """
    Calculate the long range Coulomb interactions at each coordinate.

    Parameters:
    - OLr1: Overlap between wavefunction 1 and reference wavefunction at every coordinate.
    - OLr2: Overlap between wavefunction 2 and reference wavefunction at every coordinate.
    - dist: Distance matrix.
    - invdist: Inverse distance matrix.

    Returns:
    Array of zeroth order exchange interactions at each coordinate.

    """

    return np.sum(np.outer(OLr1,OLr2)*invdist, dtype=np.clongdouble)

def K_SR(WF1, WF2, WF3, WF4, D_SRK):
    I = 0+0J
    for i in prange(3):
        #I += WF_sigma_WF[:,i]*WF_J_WF[:,i]
        I += sigma_expectation(WF1, WF2)[:,i] * J_expectation(WF3, WF4)[:,i]
    return -1*np.sum(D_SRK*I)

#@profile
def K_LR_0(OLr1, OLr2, invdist):
    """
    Calculate the zeroth order exchange interactions at each coordinate.

    Parameters:
    - OLr1: Overlap between wavefunction 1 and reference wavefunction at every coordinate.
    - OLr2: Overlap between wavefunction 2 and reference wavefunction at every coordinate.
    - dist: Distance matrix.
    - invdist: Inverse distance matrix.

    Returns:
    Array of zeroth order exchange interactions at each coordinate.

    """

    return -1*np.sum(np.outer(OLr1,OLr2)*invdist, dtype=np.clongdouble)

#@profile
@jit(nopython=True, parallel=True)
def K_LR_1(OLr1, OLr2, Pr1, Pr2, dr_vec, invdist):
    """
    Calculate the first order exchange interactions at each coordinate.

    Parameters:
    - OLr1: Overlap between wavefunction 1 and reference wavefunction.
    - OLr2: Overlap between wavefunction 2 and reference wavefunction.
    - Pr1: Momentum matrix elements for wavefunction 1.
    - Pr2: Momentum matrix elements for wavefunction 2.
    - invdist: Inverse distance matrix.

    Returns:
    Array of first order exchange interactions at each coordinate.

   
    I=0
    for xi in range(3):
        I -= np.sum(((np.outer(Pr1[:,xi], OLr2) - np.outer(OLr1, Pr2[:,xi]))* dr_vec[xi,:,:])*invdist**3)
    return I
     
    """
    I = 0+0J
    #OLr1 = np.ones(OLr1.shape(0))
    #OLr2 = np.ones(OLr1.shape(0))
    for xi in prange(3):
        I += np.sum((np.outer(Pr1[:,xi], OLr2) - np.outer(OLr1, Pr2[:,xi])) * dr_vec[xi,:,:] * invdist**3)#, dtype=np.clongdouble)
    return I


#@profile
@jit(nopython=True, parallel=True)
def K_LR_2_DD(Pr1, Pr2, dr_vec, invdist):
    I = 0+0J
    for xi in range(3):
        Pr1_xi = Pr1[:,xi]
        #I += (dist**2-3*dr_vec[xi,:,:]**2)*np.outer(Pr1[:,xi],Pr2[:,xi])
        I -= np.sum(np.outer(Pr1_xi,Pr2[:,xi])*invdist**3)#, dtype=np.clongdouble) #- 3*dr_vec[xi,:,:]**2 * np.outer(Pr1_xi,Pr2[:,xi])*invdist**5)
        for xj in prange(3):
            #if xi != xj:
            I += np.sum(3*dr_vec[xi,:,:]*dr_vec[xj,:,:]*np.outer(Pr1_xi,Pr2[:,xj])*invdist**5)#, dtype=np.clongdouble)
    return I

