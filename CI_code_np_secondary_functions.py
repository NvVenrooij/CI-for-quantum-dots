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

def OL_compare(WF1dat,WF2dat):
    return np.sum( np.conj(WF1dat) * WF2dat, axis=1)

def OL_simple(WF1dat, WF2dat):
    OL_array = np.zeros(WF1dat.shape[0])
    for i in range(WF1dat.shape[0]):
        OL_array[i] = np.dot(np.conj(WF1dat[i,:]), WF2dat[i,:])
    return OL_array

def P_compare(WF1dat, WF2dat, d_xyz):
    MMEs = np.zeros((WF1dat.shape[0],3),dtype=complex)
    for i in range(3):
        MMEs[:,i] = np.einsum('ki,ij,kj->k', np.conj(WF1dat), d_xyz[i], WF2dat)
    return MMEs

def P_simple(WF1dat, WF2dat, d_xyz):
    MMEs = np.zeros((WF1dat.shape[0],3),dtype=np.cdouble)
    for i in range(WF1dat.shape[0]):
        for j in range(3):
            MMEs[i,j] = np.dot(np.conj(WF1dat[i,:]),np.matmul(d_xyz[j,:,:],WF2dat[i,:]))
    return MMEs

def sigma_simple(WF1dat, WF2dat, sigma_xyz):
    WF_sigma_WF = np.zeros((WF1dat.shape[0],3),dtype=np.cdouble)
    for i in range(WF1dat.shape[0]):
        for j in range(3):
            WF_sigma_WF[i,j] = np.dot(np.conj(WF1dat[i,:2]), np.matmul(sigma_xyz[j], WF2dat[i,:2]))
    
    return WF_sigma_WF

def J_simple(WF1dat, WF2dat, J_xyz):
    WF_sigma_WF = np.zeros((WF1dat.shape[0],3),dtype=np.cdouble)
    for i in range(WF1dat.shape[0]):
        for j in range(3):
            WF_sigma_WF[i,j] = np.dot(np.conj(WF1dat[i,2:6]), np.matmul(J_xyz[j], WF2dat[i,2:6]))
    
    return WF_sigma_WF

def K_LR_1_simpler(WF1_dat, WF2_dat, dipole, coo):
    I = 0 
    l = WF1_dat.shape[0]
    
    # Expanded to 2D arrays for OLr1 and OLr2
    OLr1_arr = np.zeros((l, l), dtype=complex)
    OLr2_arr = np.zeros((l, l), dtype=complex)
    
    # MME arrays
    Pr1_arr = np.zeros((l, l, 3), dtype=complex)
    Pr2_arr = np.zeros((l, l, 3), dtype=complex)

    invdist_arr = np.zeros((l, l), dtype=complex)
    dr_arr = np.zeros((3, l, l), dtype=complex)

    for i in range(l):
        OLr1 = np.dot(np.conj(WF1_dat[i,:]), WF2_dat[i,:])
        OLr1_arr[i, i] = OLr1

        for j in range(l):
            if i != j:
                OLr2 = np.dot(np.conj(WF2_dat[j,:]), WF1_dat[j,:])
                OLr2_arr[i, j] = OLr2
                
                invdist_arr[i,j] = 1 / np.sqrt((coo[i,0] - coo[j,0])**2 + (coo[i,1] - coo[j,1])**2 + (coo[i,2] - coo[j,2])**2)

                for k in range(3):
                    dr_arr[k,i,j] = coo[i, k] - coo[j, k]
                    
                    # Compute MMEs and store them
                    Pr1_arr[i, j, k] = np.dot(np.conj(WF1_dat[i, :]), np.dot(dipole[k], WF2_dat[i, :]))
                    Pr2_arr[i, j, k] = np.dot(np.conj(WF2_dat[j, :]), np.dot(dipole[k], WF1_dat[j, :]))

                    I += (Pr1_arr[i, j, k] * OLr2 - OLr1 * Pr2_arr[i, j, k]) * dr_arr[k,i,j] * invdist_arr[i,j]**3

    return I, OLr1_arr, OLr2_arr, Pr1_arr, Pr2_arr, invdist_arr, dr_arr  # Return arrays along with I

def K_LR_1_simple(OLr1, OLr2, Pr1, Pr2, dr_vec, invdist,l):
    I = 0 
    for i in range(l):
        for j in range(l):
            for k in range(3):
                I += (Pr1[i,k]*OLr2[j]-OLr1[i]*Pr2[j,k]) * dr_vec[k,i,j] * invdist[i,j]**3
                #print("inv:",invdist[i,k])
                #print("dr:",dr_vec[k,i,j])
    return I

def compare_K_LR_1(WF1, WF2, nr_iterations):
    dr_vec, invdr = compute_distance_matrices(WF1.coo)
    OLr1 = OL(WF1,WF2)
    OLr2 = OL(WF2,WF1)
    Pr1 = P(WF1,WF2)
    Pr2 = P(WF2,WF1)
    #A_old = np.zeros(nr_iterations,dtype=np.cdouble)
    A_new = np.zeros(nr_iterations,dtype=np.cdouble)
    for i in prange(nr_iterations):
        A_new[i] = K_LR_1(OLr1[:i], OLr2[:i], Pr1[:i,:], Pr2[:i,:], dr_vec[:,:i,:i], invdr[:i,:i])
    A_old = K_LR_1_simpler(WF1.data[:nr_iterations], WF2.data[:nr_iterations], WF1.dipole_matrices, WF1.coo[:nr_iterations])
    #plt.plot(np.arange(nr_iterations),A_simple, label="new")
    plt.plot(np.arange(nr_iterations),A_old, label="old")
    plt.plot(np.arange(nr_iterations),A_new, label="newest")
    plt.legend()
    plt.title("K_LR_1 old vs new with increasing system size")
    plt.xlabel("# iterations")
    plt.ylabel("size K_LR_1")
    plt.show()
    return A_new,A_old

def compare_OLs_and_Ps(WF1,WF2):
    P1_old = np.zeros((WF1.data.shape[0],3), dtype=complex)
    OL1_old = np.zeros(WF1.data.shape[0], dtype=complex)
    P2_old = np.zeros((WF1.data.shape[0],3), dtype=complex)
    OL2_old = np.zeros(WF1.data.shape[0], dtype=complex)
    for i in range(WF1.data.shape[0]):
        OL2_old[i] = np.dot(np.conj(WF2.data[i,:]),WF1.data[i,:])
        OL1_old[i] = np.dot(np.conj(WF1.data[i,:]),WF2.data[i,:])
        for k in range(3):
            P1_old[i,k] = np.dot(np.conj(WF1.data[i,:]),np.matmul(WF1.dipole_matrices[k],WF2.data[i,:]))
            P2_old[i,k] = np.dot(np.conj(WF2.data[i,:]),np.matmul(WF1.dipole_matrices[k],WF1.data[i,:]))
    P1_new = P(WF1,WF2)
    P2_new = P(WF2,WF1)
    OL1_new = OL(WF1,WF2)
    OL2_new = OL(WF2,WF1)
    plt.plot(OL1_new, label = "new")
    plt.plot(OL1_old, label = "old")
    plt.legend()
    plt.show()
    plt.figure(figsize=(8, 8))  # Create new figure with a specified size
    for i in range(3):
        plt.subplot(3, 1, i+1)  # 3 rows, 1 column, subplot index
        plt.plot(P1_new[:, i], label="new")
        plt.plot(P1_old[:, i], label="old")
        plt.legend()
        plt.title(f"P component {i+1}")
    plt.tight_layout()
    plt.show()
    return OL1_new,P2_new,OL1_old,P2_old

