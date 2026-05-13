import numpy as np
from numpy import linalg as LA
import time

from basis import get_basis_shape, represent_in_terms_of_ground
from interactions import (
    compute_distance_matrices,
    OL,
    P,
    J_LR,
    K_SR,
    K_LR_0,
    K_LR_1,
    K_LR_2_DD,
)

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

    return H_0_std, H_J_LR_std, H_K_SR_std, H_K_LR_0_std, H_K_LR_1_std, H_K_LR_2_DD_std




#@profile
def CI_exciton_large(basis, framework="cv", SR=True, J=True, BM0=True, BM1=True, DD=True):


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
                
                if J == True:
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

    return H_0_std, H_J_LR_std, H_K_SR_std, H_K_LR_0_std, H_K_LR_1_std, H_K_LR_2_DD_std 



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

def calculate_energies_and_oscillator_strengths(H_CI, CI_basis, fock_space):

    eigenenergies, CIcs = LA.eig(H_CI)
    g, es, signs = represent_in_terms_of_ground(CI_basis, fock_space)
    o_strengths = dipole_transitions(CI_basis, CIcs, signs)
    e_os = list(zip(eigenenergies,o_strengths))

    return CIcs, e_os
