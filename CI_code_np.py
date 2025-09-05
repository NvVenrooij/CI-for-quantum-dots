#%%
import os
import numpy as np
#np.show_config()
from numpy import linalg as LA
import CI_code_np_primary_functions as CI_code_np_primary_functions
from CI_code_np_primary_functions import *
import importlib
importlib.reload(CI_code_np_primary_functions)
import time
import multiprocessing as mp
from line_profiler import LineProfiler
import sys



#Notes 27-09-23:
#the CI function works for small dots only. once a smaller dot gets computed the code runs into problems 
#since the WF data is being divided into two or more arrays. The way this is implemented at the moment 
#is problematic for for H_0 and H_K_SR, H_0 is being computed and added every time, resulting in
#a final exciton energy that is aprrox #chunks * #WFs higher than it should be. H_K_SR should only be calculated
#for on site interactions and therefore only when i==j. 

#@profile
def main():

    path = "/Users/nielsvanvenrooij/Library/CloudStorage/Dropbox/Mac/Documents/phD/Projects/CI for high-symmetry qds/spherical_dots_paper_2/InAs_dots/SphericalHarmonic_series/Rxyz_3_SHA_V_04"
    electron_file_name = "0.2483927480530eV.txt"
    hh_file_name = None
    lh_file_name = '0-1.190970497010eV_LH.txt'
    grid_spacing = 0.5
    InAs = Material(name="InAs", Ep=21.5, Eg=0.417, Delta_SO=0.39, Delta_SRK=0.19723521167947522, ep=15.15, a0=0.0529177210903)
    fw = "cv"
    mem_buff = 0.5
    save = True
 

    if electron_file_name != None:
        E = Wavefunction.from_file(path + "/", electron_file_name, grid_spacing, InAs, "e")
    else:
        E = None
    if hh_file_name != None:
        HH = Wavefunction.from_file(path + "/", hh_file_name, grid_spacing, InAs, "hh")
    else:
        HH = None
    if lh_file_name != None:
        LH = Wavefunction.from_file(path + "/", lh_file_name, grid_spacing, InAs, "lh")
    else:
        LH = None

    wfs = (E, HH, LH)

    new_len, old_len = compute_truncated_array_length(E.data[:,0], buffer_ratio = mem_buff)

    start = time.time()

    # Create CI_basis:
    CI_basis, fock_space = create_CI_basis(wfs, print_basis=True)
    g, es, signs = represent_in_terms_of_ground(CI_basis, fock_space)
    l_basis = get_basis_shape(CI_basis)[0]
    H_CI = np.zeros((l_basis,l_basis),dtype=np.cdouble)

    if old_len > new_len:
        number_chunks = old_len//new_len+1
        for i in range(len(CI_basis)):
            CI_basis[i][0] = Wavefunction.split_wf(CI_basis[i][0],number_chunks)
            CI_basis[i][1] = Wavefunction.split_wf(CI_basis[i][1],number_chunks)
        hams = np.cdouble(CI_exciton_large(CI_basis, framework=fw, SR=True, BM0=True, BM1=True, DD=True))
        print("SR:", repr(hams[2]))
        print("SR_eig:", LA.eig(hams[2])[0])
        print("LR0:", repr(hams[3]))
        print("LR0_eig:", LA.eig(hams[3])[0])
        print("LR1:", repr(hams[4]))
        print("LR1_eig:", LA.eig(hams[4])[0])
        print("LR2DD:", repr(hams[5]))
        print("LR2DD_eig:", LA.eig(hams[5])[0])
        #H_CI += hams[0] + hams[1] + hams[2] + hams[3] + hams[4] + hams[5]
    
    else: 
        hams = np.cdouble(CI_exciton_small(CI_basis, framework=fw, SR=True, BM0=True, BM1=True, DD=True))
        #H_CI += hams[0] + hams[1] + hams[2] + hams[3] + hams[4] + hams[5]
            
    CI_basis, fock_space = create_CI_basis(wfs, print_basis=True)
    ham_dict = {'J_LR+K_SR': hams[0]+ hams[1] + hams[2],
                'J_LR+K_LR0': hams[0]+ hams[1] + hams[3],
                'J_LR+K_LR1': hams[0]+ hams[1] + hams[4],
                'J_LR+KLR2': hams[0]+ hams[1] + hams[5],
                'band_mixing': hams[0]+ hams[1] + hams[3] + hams[4],
                'SR+DD': hams[0]+ hams[1] + hams[2] + hams[5],
                'full': hams[0]+ hams[1] + hams[3] + hams[4] + hams[5]}
    e_os_dict = {}
    CIcs_dict = {}

    for key, H_CI in ham_dict.items():
        CIcs_dict[key], e_os_dict[key] = calculate_energies_and_oscillator_strengths(H_CI, CI_basis, fock_space)

    #eigenenergies, CIcs = LA.eig(H_CI)
    #g, es, signs = represent_in_terms_of_ground(CI_basis, fock_space)
    #o_strengths = dipole_transitions(CI_basis, CIcs, signs)

    #e_os = list(zip(eigenenergies,o_strengths))
    print('(Energies, Oscillator strenghts:', e_os_dict)

    print(time.time()-start, "seconds")

    if save == True:
        save_data(E, hams, H_CI, e_os_dict, CIcs_dict, path, HH, LH)

if __name__ == "__main__":
    main()



#%%

        

