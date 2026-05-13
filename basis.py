
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
