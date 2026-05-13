from scipy.spatial.distance import cdist
from functools import cache
import pickle


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
            "H_J_LR":hams[1],
            "H_SR":hams[2],
            "H_LR0":hams[3],
            "H_LR1":hams[4],
            "H_LR2DD":hams[5],
            "H_CI":H_CI,
            "eigenenergies_and_oscillator_strengths":e_os,
            "CI-coefficients":CIcs
    }

    with open(path + '/' + 'CI_calculation' + string + '.pkl', 'wb') as file:
        pickle.dump(pickle_data, file)