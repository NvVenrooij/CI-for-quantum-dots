#%%
import numpy as np
from numpy import linalg as LA
import scipy as sc
from scipy.spatial.distance import cdist
import psutil
import math
import pandas as pd
#import numexpr as ne
from line_profiler import LineProfiler
import time
from numba import jit, prange
import matplotlib.pyplot as plt
from functools import cache
import pickle
import json


class Wavefunction:
    def __init__(self, data, coo, energy, grid_spacing, material=None, band_label=None):
        self.data = np.array(data, dtype=np.clongdouble)  
        self.coo = np.array(coo, dtype=np.longdouble)     
        self.energy = np.longdouble(energy)               
        self.grid_spacing = np.longdouble(grid_spacing)   
        self.material = material
        self.band_label = band_label
        self.Delta_SRK = material.Delta_SRK*grid_spacing**(-3)
        self.dipole_matrices = material.dipole_matrices
        self.pauli_matrices = self.get_pauli_matrices()
        self.J_matrices = self.get_J_matrices()
        
    def time_reverse(self):
            modified_data = self._calculate_time_reverse(self.data)
            modified_band_label = "T" + self.band_label
            return Wavefunction(modified_data, self.coo, self.energy, self.grid_spacing, self.material, modified_band_label)
    
    @classmethod
    def from_file(cls, file_directory, wf_file, grid_spacing, material=None, band_label=None):
        wf_array = cls.import_wfdata(file_directory + wf_file)
        data, coo = cls.reshape_wfdata(wf_array)
        energy = cls.get_energy(wf_file)
        return cls(data, coo, energy, grid_spacing=grid_spacing, material=material, band_label=band_label)

    @classmethod
    def split_wf(cls, wf, num_chunks):
        """
        Split the WF object into multiple separate WF objects, each containing a chunk of the original data.

        Parameters:
        - wf: The original Wavefunction object.
        - num_chunks: The number of chunks to split the data into.

        Returns:
        A list of Wavefunction objects, each containing a portion of the original WF.data.
        """
        if num_chunks == 1:
            num_chunks = 2

        len_data = len(wf.data)
        chunk_size = len_data // num_chunks
        remainder = len_data % num_chunks

        wf_chunks = np.array_split(wf.data, num_chunks)
        coo_chunks = np.array_split(wf.coo, num_chunks)

        # Find the maximum chunk length
        max_len = max(len(chunk) for chunk in wf_chunks)

        # Pad all chunks to the same length
        for i in range(num_chunks):
            if len(wf_chunks[i]) < max_len:
                num_zeros = max_len - len(wf_chunks[i])
                wf_padding = np.zeros((num_zeros,) + wf.data.shape[1:], dtype=wf.data.dtype)
                coo_padding = np.zeros((num_zeros,) + wf.coo.shape[1:], dtype=wf.coo.dtype)
                wf_chunks[i] = np.concatenate((wf_chunks[i], wf_padding), axis=0)
                coo_chunks[i] = np.concatenate((coo_chunks[i], coo_padding), axis=0)

        return [cls(wf_chunks[i], coo_chunks[i], wf.energy, wf.grid_spacing, wf.material, wf.band_label)
                for i in range(num_chunks)]

    @classmethod
    def get_pauli_matrices(self):
        sx = np.array([[0,0.5],[0.5,0]],dtype=np.clongdouble)
        sy = np.array([[0,0.5*1J],[-1J*0.5,0]],dtype=np.clongdouble)
        sz = np.array([[-0.5,0],[0,0.5]],dtype=np.clongdouble)
        return np.array([sx,sy,sz])

    @classmethod
    def get_J_matrices(self):

        s32 = np.longdouble(np.sqrt(3)*0.5)
        Is32 = 1J*np.longdouble(np.sqrt(3)*0.5)

        Jx = np.array([[0, -s32, 0, 1],[-s32,0,0,0],[0,0,0,-s32],[1,0,-s32,0]],dtype=np.clongdouble)
        Jy = np.array([[0,-Is32,0,-1J],[Is32,0,0,0],[0,0,0,-Is32],[1J,0,Is32,0]],dtype=np.clongdouble)
        Jz = np.array([[0.5,0,0,0],[0,1.5,0,0],[0,0,-1.5,0],[0,0,0,-0.5]],dtype=np.clongdouble)
        return np.array([Jx,Jy,Jz])

    @staticmethod
    def get_energy(wf_file):
        if "_LH" in wf_file:
            energy = wf_file.replace("_LH","")
        if "_HH" in wf_file:
            energy = wf_file.replace("_HH","")
        if "-" in wf_file:
            energy = energy.replace("0-","")
            energy = -1*float(energy.replace("eV.txt",""))
        else:
            energy = float(wf_file.replace("eV.txt",""))
        return energy

    @staticmethod
    def import_wfdata(file_directory):
        """Opens raw data files computed by the code on the argon cluster
        and transfers it to a numpy array of appropriate size."""
        with open(file_directory, "r") as f:
            for line in range(5):
                next(f)
            for line in f:
                wfdat = f.read()
                wfdat = wfdat.replace("(", "")
                wfdat = wfdat.replace(")", "")
                wfdat = wfdat.replace(",", " ")
                wfdat = wfdat.replace("\n", " ")
                wfdat = wfdat.replace("\t", " ")
                wfdat = wfdat.split()
                wfdat = [np.double(i) for i in wfdat]
                n_rows = len(wfdat) // 19
                wfmatrix = np.reshape(wfdat, (n_rows, 19))
        return wfmatrix

    @staticmethod
    def reshape_wfdata(wf_array):
        coo = np.array(wf_array[:, :3], dtype=np.longdouble)
        wf = wf_array[:, 3:]
        wf = wf * np.tile([1, 1j], (np.shape(wf_array)[0], 8))
        wf = wf[:, 0::2] + wf[:, 1::2]
        return np.array(wf, dtype=np.clongdouble), coo
    
    @staticmethod
    def _calculate_time_reverse(wf_array):
        # Perform the time-reversal operation on the wf_array and return the modified array
        TR = np.zeros((np.shape(wf_array)[0], 8), dtype=np.clongdouble)

        TR[:, 0] = np.conj(wf_array[:, 1])
        TR[:, 1] = -np.conj(wf_array[:, 0])
        TR[:, 2] = np.conj(wf_array[:, 5])
        TR[:, 3] = -np.conj(wf_array[:, 4])
        TR[:, 4] = np.conj(wf_array[:, 3])
        TR[:, 5] = -np.conj(wf_array[:, 2])
        TR[:, 6] = np.conj(wf_array[:, 7])
        TR[:, 7] = -np.conj(wf_array[:, 6])

        TR = 1 / np.sqrt(np.sum(np.sum(TR * np.conj(TR), 1), 0)) * TR

        return TR

    
class Material:
    def __init__(self, name, Ep, Eg, Delta_SO, Delta_SRK, ep, a0):
        self.name = name
        self.Kane_param = Ep
        self.Band_gap = Eg
        self.Delta_SO = Delta_SO
        self.Delta_SRK = Delta_SRK
        self.epsilon_inf = ep
        self.lattice_const = a0
        self.dipole_matrices = self.calculate_dipole_matrices(Delta_SO, Eg)  # Calculate the operator matrix

    @classmethod
    def calculate_dipole_matrices(cls, Delta_SO, Eg):
        Delta_tilde = 1/(1+Delta_SO/Eg) #Due to lack of time however this is not yet implemented.

        # nonzero elements of numerical dipole matrix 
        
        Dx02 = -1J*1/np.sqrt(6)
        Dx04 = -1J*1/np.sqrt(2)
        Dx07 = Delta_tilde * -1J * 1/np.sqrt(3)
        Dx13 = -1J*-1/np.sqrt(2)
        Dx15 = -1J*-1/np.sqrt(6)
        Dx16 = Delta_tilde * -1J*1/np.sqrt(3)
        Dx20 = np.conj(Dx02)
        Dx31 = np.conj(Dx13)
        Dx40 = np.conj(Dx04)
        Dx51 = np.conj(Dx15)
        Dx61 = np.conj(Dx16)
        Dx70 = np.conj(Dx07)

        Dx = np.array([[0, 0, Dx02, 0, Dx04, 0, 0, Dx07],               
                        [0, 0, 0, Dx13, 0, Dx15, Dx16, 0],
                        [Dx20, 0, 0, 0, 0, 0, 0, 0],
                        [0, Dx31, 0, 0, 0, 0, 0, 0],
                        [Dx40, 0, 0, 0, 0, 0, 0, 0],
                        [0, Dx51, 0, 0, 0, 0, 0, 0],
                        [0, Dx61, 0, 0, 0, 0, 0, 0],
                        [Dx70, 0, 0, 0, 0, 0, 0, 0]],dtype=np.clongdouble)              
        
        Dy02 = -1J*1J/np.sqrt(6)
        Dy04 = -1J*-1J/np.sqrt(2)
        Dy07 = Delta_tilde * -1J*1J/np.sqrt(3)
        Dy13 = -1J*-1J/np.sqrt(2)
        Dy15 = -1J*1J/np.sqrt(6)
        Dy16 = Delta_tilde * -1J*-1J/np.sqrt(3)
        Dy20 = np.conj(Dy02)
        Dy31 = np.conj(Dy13)
        Dy40 = np.conj(Dy04)
        Dy51 = np.conj(Dy15)
        Dy61 = np.conj(Dy16)
        Dy70 = np.conj(Dy07)


        Dy = np.array([[0, 0, Dy02, 0, Dy04, 0, 0, Dy07],
                        [0, 0, 0, Dy13, 0, Dy15, Dy16, 0],
                        [Dy20, 0, 0, 0, 0, 0, 0, 0],
                        [0, Dy31, 0, 0, 0, 0, 0, 0],
                        [Dy40, 0, 0, 0, 0, 0, 0, 0],
                        [0, Dy51, 0, 0, 0, 0, 0, 0],
                        [0, Dy61, 0, 0, 0, 0, 0, 0],
                        [Dy70, 0, 0, 0, 0, 0, 0, 0]],dtype=np.clongdouble)

        Dz05 = -1J*-np.sqrt(2/3)
        Dz06 = Delta_tilde * -1J*-1/np.sqrt(3)
        Dz12 = -1J*-np.sqrt(2/3)
        Dz17 = Delta_tilde * -1J*1/np.sqrt(3)
        Dz21 = np.conj(Dz12)
        Dz50 = np.conj(Dz05)
        Dz60 = np.conj(Dz06)
        Dz71 = np.conj(Dz17)      

        Dz = np.array([[0, 0, 0, 0, 0, Dz05, Dz06, 0],
                        [0, 0, Dz12, 0, 0, 0, 0, Dz17],
                        [0, Dz21, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [Dz50, 0, 0, 0, 0, 0, 0, 0],
                        [Dz60, 0, 0, 0, 0, 0, 0, 0],
                        [0, Dz71, 0, 0, 0, 0, 0, 0]],dtype=np.clongdouble)

        return np.array([Dx,Dy,Dz]) 


def compute_truncated_array_length(arr, buffer_ratio=0.5):
    """
    Compute the maximum length of a truncated array based on available memory and buffer ratio.

    Parameters:
    - arr1: First array.
    - arr2: Second array.
    - max_memory: Maximum available memory (default: None).
    - buffer_ratio: Ratio of allocated memory to maximum memory (default: 0.8).

    Returns:
    - max_array_length: Maximum length of the truncated array.

    """

    # Compute maximum amount of memory available
    # Depending on the buffer ratio, the allocated memory <= max memory
   
    max_memory = psutil.virtual_memory().available
    available_memory = int(max_memory * buffer_ratio)

    # Determine the size that the arrays will be
    old_array_length = arr.shape[0]
    required_memory_outer_product_array = arr.itemsize * arr.size * arr.size
    required_memory_dr_vec_array = required_memory_outer_product_array*3/2 #division by 2 because it is not a np.clongdouble 128 but float 64 
    required_memory_dist_array = required_memory_dr_vec_array*0.5

    # Print the computed values for analysis
    print("Maximum available memory =", max_memory/ (1024 ** 3), "GB")
    print("Available memory (including buffer) =", available_memory/ (1024 ** 3), "GB")
    print("Required memory for outer product wf array =", required_memory_outer_product_array/ (1024 ** 3), "GB")
    print("Required memory for dr_vec array =", required_memory_dr_vec_array/ (1024 ** 3), "GB")
    print("Required memory for dist array =", required_memory_dist_array/ (1024 ** 3), "GB")

    required_memory = required_memory_outer_product_array + required_memory_dr_vec_array + required_memory_dist_array
    print("Required Memory: ", required_memory/ (1024 ** 3), "GB")


    # Determine the maximum chunk size by taking the minimum value between
    # available_memory and the memory required for the 2D array
    if available_memory <= required_memory:
        max_arr_size = int(available_memory*(required_memory_dist_array/(required_memory_outer_product_array+required_memory_dr_vec_array+required_memory_dist_array))) #the arrays are all the same size but 
        print("max arr size =", max_arr_size)
        max_number_elements = int(max_arr_size / arr.itemsize)
        print("Not enough memory")
        print("Max array memory =", max_arr_size/ (1024 ** 3), "GB")
        print("Max number of elements per chunk =", max_number_elements)

        # Calculate the maximum array length based on the maximum number of elements per chunk
        print("Old array length =", old_array_length)
        new_array_length = math.isqrt(max_number_elements)
        print("New array length =", new_array_length)
        #max_array_length = int(integer_square_mne / 8)  # divided by 8 because array is l x 8

        return new_array_length, old_array_length  
    
    else:
        max_arr_size = required_memory_outer_product_array
        max_number_elements = int(arr.size * arr.size)
        new_array_length = old_array_length
        print("Enough spare memory")
        print("Old array length =", old_array_length)
        print("New array length =", new_array_length)

        return new_array_length, old_array_length





def slice_arrays(arr, coo, trunc_array_length):
    """
    Slice arrays into smaller chunks based on the specified truncation array length.

    Parameters:
    - WF: Array to be sliced.
    - coo: Coordinate array to be sliced.
    - trunc_array_length: Length of the truncated arrays.

    Returns:
    - WF_arrays: list containing sliced arrays.
    - coos: Sliced arrays of coo.

    """

    # Compute the quotient and remainder of WF.size divided by trunc_array_length
    quotient, remainder = divmod(arr.size, trunc_array_length)

    # Compute the split points to split WF and coo arrays into smaller chunks
    #split_points = [i * trunc_array_length for i in np.arange(1, quotient, 1)]

    # Split WF array into smaller chunks based on the split points
    #WF_arrays = np.split(WF, split_points, axis=0)
    arrays = np.array_split(arr, len(arr)// trunc_array_length, axis=0)

    # Flatten WF array
    #for i in range(quotient):
    #    WF_arrays[i] = WF_arrays[i].flatten()

    # Split coo array into smaller chunks based on the split points
    coos = np.array_split(coo, len(coo)//trunc_array_length, axis=0)

    return arrays, coos


def compare_WFs(WF):
    if WF.band_label == "e":
        return (0, WF.energy)
    if WF.band_label == "Te":
        return (1, WF.energy)
    if WF.band_label == "hh":
        return (2, WF.energy)
    if WF.band_label == "Thh":
        return (3, WF.energy)
    if WF.band_label == "lh":
        return (4, WF.energy)
    if WF.band_label == "Tlh":
        return (5, WF.energy)

def create_CI_basis(WFs, print_basis):

    """
    Order wavefunctions in the tuple depending on the band_label attribute belonging to the WF object.
    The correct order depends on the number of input wavefunctions but always follows the following scheme:
    The first wavefunction is always the LOWEST ENERGY ELECTRON, the second is always its TIME REVERSE.
    Depending on the number of electron wavefunctions, higher energy electron WFs will be 3rd, 4th, etc...
    The LOWEST ENERGY HOLE wavefunction and its TIME REVERSE, will be added after the electrons.
    """

    e_basis = []
    h_basis = []

    # Compute time reverses:

    for wf in range(len(WFs)):
        if WFs[wf] != None:
            if 'e' in WFs[wf].band_label:
                e_basis += [WFs[wf]] + [WFs[wf].time_reverse()]
            else:
                h_basis += [WFs[wf]] + [WFs[wf].time_reverse()]

    e_basis = sorted(e_basis, key = compare_WFs)
    h_basis = sorted(h_basis, key = compare_WFs)
    fock_space = e_basis + h_basis

    if print_basis == True:
        print("e_basis:")
        for i in range(len(e_basis)):
            print(e_basis[i].band_label, e_basis[i].energy)
        
        print("h_basis:")
        for i in range(len(h_basis)):
            print(h_basis[i].band_label, h_basis[i].energy)
    
    CI_basis = []
    if print_basis  == True:
        print("CI_basis:")
    for i in range(len(e_basis)):
        for j in range(len(h_basis)):
            CI_basis.append([e_basis[i],h_basis[j]])
            if print_basis == True:
                print(CI_basis[len(h_basis)*i+j][0].band_label, CI_basis[len(h_basis)*i+j][1].band_label)

    return CI_basis, fock_space


def represent_in_terms_of_ground(CI_basis, fock_space):

    g = []
    for i in range(len(fock_space)):
        if 'e' in fock_space[i].band_label:
            g.append(0)
        else:
            g.append(1)

    es = []
    signs = []
    for i in range(len(CI_basis)):
        e = g.copy()
        for k in range(len(fock_space)):
            electron_found = False
            hole_found = False  
            for j in range(2):
                if CI_basis[i][j].band_label == fock_space[k].band_label:
                    if 'e' in fock_space[k].band_label:
                        electron_found = True
                    elif 'h' in fock_space[k].band_label:
                        hole_found = True
                    break
            if electron_found:
                e[k] = 1
            elif hole_found:  
                e[k] = 0
        es.append(e)
        sign = 1
    
        for i, (g_val, e_val) in enumerate(zip(g, e)):
            if g_val == 1 and e_val == 0:  # Annihilation
                sign *= (-1) ** sum(g[:i])
            elif g_val == 0 and e_val == 1:  # Creation
                sign *= (-1) ** sum(g[:i+1])
        signs.append(sign)
    
    return g, es, signs


def get_basis_shape(lst):
    if not isinstance(lst, list):
        return []
    return [len(lst)] + get_basis_shape(lst[0])


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


##@profile
def CI_exciton_small(basis, framework="cv", SR=True, J=True, BM0=True, BM1=True, DD=True):

    '''
    Computes the Configuration Interaction for an arbitrary number of orbitals.

    input: a minimum of 2 wavefunctions.

    output: eigenenergies of the configuration and their CI-coefficients

    note 27-09-23: Works only for small dots, not for larger dots, needs to be modified.

    note 28-09-23: try to optimise dr_vec and invdr by storing them as sparse matrices. Optionally, you can opt to store the output of 
                   K_LR_x functions in sparse matrices as well by setting the lower triangle of the matrix to zero. 
                   Don't forget in this case to multiply the coefficients with 2.
    '''


    dr_vec, invdr = compute_distance_matrices(basis[0][0].coo, basis[0][1].coo)
    l_basis =  len(basis) 
    H_0 = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    for i in range(l_basis):
        H_0[i,i] = np.abs(basis[i][0].energy - basis[i][1].energy)
    H_K_SR = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_K_LR_0 = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_K_LR_1 = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_K_LR_2_DD = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_K_LR_2_BM = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_J_LR = np.zeros((l_basis,l_basis), dtype= np.clongdouble)

    OLr1 = np.zeros((l_basis, l_basis, len(basis[0][0].data)), dtype= np.clongdouble)
    OLr2 = np.zeros((l_basis, l_basis, len(basis[0][0].data)), dtype= np.clongdouble)
    MMEr1 = np.zeros((l_basis, l_basis, len(basis[0][0].data), 3), dtype= np.clongdouble)
    MMEr2 = np.zeros((l_basis, l_basis, len(basis[0][0].data), 3), dtype= np.clongdouble)

    D_SRK = np.clongdouble(basis[0][0].Delta_SRK)
    a0_in_nm = np.clongdouble(basis[0][0].material.lattice_const)
    ep = np.clongdouble(basis[0][0].material.epsilon_inf)
    Ep = np.clongdouble(basis[0][0].material.Kane_param)
    Eg = np.clongdouble(basis[0][0].material.Band_gap)

    # implement cache list?
    # paralellise H_K_LR_x functions in the inner for loop (each of these functions can run on 1 core)
    # additionally, each outer product in the H_K_LR_x functions can be computed on 1 core 
            
    for i in range(l_basis):

        if J == True:
                OLr1_ee = OL(basis[i][0],basis[i][0])
                OLr2_hh = OL(basis[i][1],basis[i][1])
                H_J_LR[i,i] = J_LR(OLr1_ee, OLr2_hh, invdr)
        
        for j in range(l_basis):
            print("percentage inner loop:", (l_basis*i+j)/(l_basis**2)*100, "%", end='\r')
            if j >= i:
                OLr1 = OL(basis[i][0],basis[j][1])
                OLr2 = OL(basis[i][1],basis[j][0])
                MMEr1 = P(basis[i][0],basis[j][1])
                MMEr2 = P(basis[i][1],basis[j][0])                  #print("Memory OL_outer:", OLr1_i.size**2*OLr1_i.itemsize/(1024**3), "GB")

                if SR == True:
                    H_K_SR[i,j] = K_SR(basis[i][0], basis[j][0], basis[i][1], basis[j][1], D_SRK)
                
                if BM0 == True: 
                    H_K_LR_0[i,j] = K_LR_0(OLr1, OLr2, invdr)

                if BM1 == True:
                    H_K_LR_1[i,j] = K_LR_1(OLr1, OLr2, MMEr1, MMEr2, dr_vec, invdr)

                if DD == True:
                    H_K_LR_2_DD[i,j] = K_LR_2_DD(MMEr1, MMEr2, dr_vec, invdr)

                #print("available memory during CI:", psutil.virtual_memory().available/ (1024 ** 3), "GB")
            else:
                H_K_SR[i,j] = np.conj(H_K_SR[j,i])
                H_K_LR_0[i,j] = np.conj(H_K_LR_0[j,i])
                H_K_LR_1[i,j] = np.conj(H_K_LR_1[j,i])
                H_K_LR_2_DD[i,j] = np.conj(H_K_LR_2_DD[j,i])
            

    Ht_in_eV = np.clongdouble(27.211386245988)
    elementary_charge_in_au = np.clongdouble(1)
    electron_mass_in_au = np.clongdouble(1)
    Coulomb_constant_in_au = np.clongdouble(1)    

    if framework == "cv":
        sign = np.clongdouble(1)
    if framework == "eh":
        sign = -np.clongdouble(1)
    
    zeroth_order_coefficient = Coulomb_constant_in_au*elementary_charge_in_au**2*a0_in_nm*1/(2*ep)*Ht_in_eV
    first_order_coefficient = Coulomb_constant_in_au*elementary_charge_in_au**2*a0_in_nm**2*1/(2*ep)*np.sqrt(Ep*Ht_in_eV/(2*Eg**2))*Ht_in_eV
    second_order_coefficient = Coulomb_constant_in_au*elementary_charge_in_au**2*a0_in_nm**3*1/(2*ep)*Ep*Ht_in_eV/(2*Eg**2)*Ht_in_eV

    H_K_SR = sign*H_K_SR
    H_J_LR = sign*zeroth_order_coefficient*H_J_LR
    H_K_LR_0 = sign*zeroth_order_coefficient*H_K_LR_0
    H_K_LR_1 = sign*first_order_coefficient*H_K_LR_1
    H_K_LR_2_DD = sign*second_order_coefficient*H_K_LR_2_DD

    print(f"\nMatrix computation completed in extended precision ({np.finfo(np.clongdouble).precision} bits)")

    # ========================================================================
    # CRITICAL: CONVERT TO STANDARD PRECISION FOR LINALG COMPATIBILITY
    # ========================================================================
    
    print("Converting to standard precision for linalg compatibility...")
    
    H_0_std = np.array(H_0, dtype=np.cdouble)
    H_K_SR_std = np.array(H_K_SR, dtype=np.cdouble)
    H_J_LR_std = np.array(H_J_LR, dtype=np.cdouble)
    H_K_LR_0_std = np.array(H_K_LR_0, dtype=np.cdouble)
    H_K_LR_1_std = np.array(H_K_LR_1, dtype=np.cdouble)
    H_K_LR_2_DD_std = np.array(H_K_LR_2_DD, dtype=np.cdouble)
    
    print(f"Converted to standard precision ({np.finfo(np.cdouble).precision} bits) - linalg compatible!")

    return H_0_std, H_K_SR_std, H_J_LR_std, H_K_LR_0_std, H_K_LR_1_std, H_K_LR_2_DD_std




#@profile
def CI_exciton_large(basis, framework="cv", SR=True, BM0=True, BM1=True, DD=True):


    '''Computes the Configuration Interaction for an arbitrary number of orbitals.

    input: a basis that consists of a minimum of 2 wavefunctions.

    output: eigenenergies of the configuration and their CI-coefficients
    
    note 28-09-23: try to optimise dr_vec and invdr by storing them as sparse matrices. Optionally, you can opt to store the output of 
                K_LR_x functions in sparse matrices as well by setting the lower triangle of the matrix to zero. 
                Don't forget in this case to multiply the coefficients with 2.

    
    '''
    
    shape_basis = get_basis_shape(basis)
    l_basis =  len(basis) 
    n_chunks = shape_basis[-1]
    H_0 = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_K_SR = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_K_LR_0 = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_K_LR_1 = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_K_LR_2_DD = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_K_LR_2_BM = np.zeros((l_basis,l_basis), dtype= np.clongdouble)
    H_J_LR = np.zeros((l_basis,l_basis), dtype= np.clongdouble)

    #wf_data_arr = np.zeros(shape_basis, dtype=np.complex64)

    for i in range(l_basis):
        H_0[i,i] = np.clongdouble(np.abs(basis[i][0][0].energy - basis[i][1][0].energy))
        #wf_data_arr

    D_SRK = np.clongdouble(basis[0][0][0].Delta_SRK)
    print("DSRK +",D_SRK)
    a0_in_nm = np.clongdouble(basis[0][0][0].material.lattice_const) 		
    ep = np.clongdouble(basis[0][0][0].material.epsilon_inf)
    Ep = np.clongdouble(basis[0][0][0].material.Kane_param)
    Eg = np.clongdouble(basis[0][0][0].material.Band_gap)

    start = time.time()

    i = 0
    j = 0

    for k in range(n_chunks):
        for l in range(n_chunks):
            print("percentage total comp:", (k*n_chunks*l_basis**2+l*l_basis**2+i*l_basis+j)/(n_chunks**2*l_basis**2)*100, "%")
            for i in range(l_basis):
                e_1 = basis[i][0][k]
                h_2 = basis[i][1][l]
                dr_vec, invdr = compute_distance_matrices(e_1.coo, h_2.coo)
                
                if J_LR == True:
                        OLr1_ee = OL(e_1,e_1)
                        OLr2_hh = OL(h_2,h_2)
                        H_J_LR[i,i] = J_LR(OLr1_ee,OLr2_hh, invdr)
                
                for j in range(l_basis):
                    h_1 = basis[j][1][k]
                    e_2 = basis[j][0][l]
                    if j >= i:
                            OLr1 = OL(e_1,h_1)
                            MMEr1 = P(e_1,h_1)

                            if SR == True:
                                if k == l:
                                    H_K_SR[i,j] += K_SR(basis[i][0][k], basis[j][0][k], basis[i][1][k], basis[j][1][k], D_SRK)

                            OLr2 = OL(h_2,e_2)
                            MMEr2 = P(h_2,e_2)

                            if BM0 == True: 
                                H_K_LR_0[i,j] += K_LR_0(OLr1, OLr2, invdr)

                            if BM1 == True:
                                H_K_LR_1[i,j] += K_LR_1(OLr1, OLr2, MMEr1, MMEr2, dr_vec, invdr)

                            if DD == True:
                                H_K_LR_2_DD[i,j] += K_LR_2_DD(MMEr1, MMEr2, dr_vec, invdr)
                            
                    else:
                        H_K_SR[i,j] = np.conj(H_K_SR[j,i])
                        H_K_LR_0[i,j] = np.conj(H_K_LR_0[j,i])
                        H_K_LR_1[i,j] = np.conj(H_K_LR_1[j,i])
                        H_K_LR_2_DD[i,j] = np.conj(H_K_LR_2_DD[j,i])
            
    end = time.time()
    print("total duration =", end-start)

    Ht_in_eV = np.clongdouble(27.211386245988)      #Hartree in eV
    elementary_charge_in_au = np.clongdouble(1)     #elementary charge in atomic units
    electron_mass_in_au = np.clongdouble(1)         #electron mass in atomic units	
    Coulomb_constant_in_au = np.clongdouble(1)      #Coulomb constant in atomic units    

    if framework == "cv":
        sign = np.clongdouble(1)
    if framework == "eh":
        sign = -np.clongdouble(1)
    
    zeroth_order_coefficient = Coulomb_constant_in_au*elementary_charge_in_au**2*a0_in_nm*1/(2*ep)*Ht_in_eV
    first_order_coefficient = Coulomb_constant_in_au*elementary_charge_in_au**2*a0_in_nm**2*1/(2*ep)*np.sqrt(Ep*Ht_in_eV/(2*Eg**2))*Ht_in_eV
    second_order_coefficient = Coulomb_constant_in_au*elementary_charge_in_au**2*a0_in_nm**3*1/(2*ep)*Ep*Ht_in_eV/(2*Eg**2)*Ht_in_eV

    H_K_SR = sign*H_K_SR
    H_J_LR = sign*zeroth_order_coefficient*H_J_LR
    H_K_LR_0 = sign*zeroth_order_coefficient*H_K_LR_0
    H_K_LR_1 = sign*first_order_coefficient*H_K_LR_1
    H_K_LR_2_DD = sign*second_order_coefficient*H_K_LR_2_DD

    print(f"\nMatrix computation completed in extended precision ({np.finfo(np.clongdouble).precision} bits)")

    # ========================================================================
    # CRITICAL: CONVERT TO STANDARD PRECISION FOR LINALG COMPATIBILITY
    # ========================================================================
    
    print("Converting to standard precision for linalg compatibility...")
    
    H_0_std = np.array(H_0, dtype=np.cdouble)
    H_K_SR_std = np.array(H_K_SR, dtype=np.cdouble)
    H_J_LR_std = np.array(H_J_LR, dtype=np.cdouble)
    H_K_LR_0_std = np.array(H_K_LR_0, dtype=np.cdouble)
    H_K_LR_1_std = np.array(H_K_LR_1, dtype=np.cdouble)
    H_K_LR_2_DD_std = np.array(H_K_LR_2_DD, dtype=np.cdouble)
    
    print(f"Converted to standard precision ({np.finfo(np.cdouble).precision} bits) - linalg compatible!")

    return H_0_std, H_K_SR_std, H_J_LR_std, H_K_LR_0_std, H_K_LR_1_std, H_K_LR_2_DD_std 



def dipole_transitions(basis, CIcs, signs):

    shape_basis = get_basis_shape(basis)
    l_basis =  shape_basis[0] 
    MMEx = np.zeros(l_basis, dtype=np.clongdouble)
    MMEy = np.zeros(l_basis, dtype=np.clongdouble)
    MMEz = np.zeros(l_basis, dtype=np.clongdouble)


    for i in range(l_basis):
        e_1 = basis[i][0]
        h_1 = basis[i][1]
        prefac = np.sqrt(e_1.material.Kane_param/(3*(e_1.energy-h_1.energy)))
        MMEs = prefac*P(h_1,e_1)
        MMEx[i] = signs[i]*np.sum(MMEs[:,0])
        MMEy[i] = signs[i]*np.sum(MMEs[:,1])
        MMEz[i] = signs[i]*np.sum(MMEs[:,2])
    
    MMEx_recom = np.zeros(l_basis, dtype=np.clongdouble)
    MMEy_recom = np.zeros(l_basis, dtype=np.clongdouble)
    MMEz_recom = np.zeros(l_basis, dtype=np.clongdouble)

    for j in range(0, l_basis, 2):
        MMEx_recom[j] = MMEx[j+1]
        MMEx_recom[j+1] = MMEx[j]
        MMEy_recom[j] = MMEy[j+1]
        MMEy_recom[1+j] = MMEy[j]
        MMEz_recom[j] = MMEz[j+1]
        MMEz_recom[1+j] = MMEz[j]

    
    px = np.matmul(np.conj(MMEx_recom),np.conj(CIcs))*np.matmul(MMEx_recom, CIcs)
    py = np.matmul(np.conj(MMEy_recom),np.conj(CIcs))*np.matmul(MMEy_recom, CIcs)
    pz = np.matmul(np.conj(MMEz_recom),np.conj(CIcs))*np.matmul(MMEz_recom, CIcs)
    
    pxyz = px + py + pz
    
    return pxyz


def save_data(E, hams, H_CI, e_os, CIcs, path, HH=None, LH=None):

    ''' Creates a JSON and PKL file that both contain relevant simulation data.
    The JSON only contains the initial paramaters and resulting eigenenergies.
    The PKL contains all data (Hamiltonians, CI coefficients, Eigenenergies, 
    Oscillator strenghts, etc..)'''

    E_energy = E.energy
    string = '_e'
    if HH != None:
        HH_energy = HH.energy
        string += '_hh'
    else:
        HH_energy = None
    if LH != None:
        LH_energy = LH.energy
        string += '_lh'
    else:
        LH_energy = None

    pickle_data = {
            "e-energy":E_energy,
            "hh-energy":HH_energy,
            "lh-energy":LH_energy,
            "grid-spacing":E.grid_spacing,
            "material":E.material.name,
            "H_SR":hams[1],
            "H_LR0":hams[2],
            "H_LR1":hams[3],
            "H_LR2DD":hams[4],
            "H_CI":H_CI,
            "eigenenergies_and_oscillator_strengths":e_os,
            "CI-coefficients":CIcs
    }

    with open(path + '/' + 'CI_calculation' + string + '.pkl', 'wb') as file:
        pickle.dump(pickle_data, file)



def calculate_energies_and_oscillator_strengths(H_CI, CI_basis, fock_space):

    eigenenergies, CIcs = LA.eig(H_CI)
    g, es, signs = represent_in_terms_of_ground(CI_basis, fock_space)
    o_strengths = dipole_transitions(CI_basis, CIcs, signs)
    e_os = list(zip(eigenenergies,o_strengths))

    return CIcs, e_os


#%%