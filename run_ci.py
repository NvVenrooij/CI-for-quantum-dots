
import time
from pathlib import Path

import numpy as np
from numpy import linalg as LA

from material import Material
from wavefunction import Wavefunction
from basis import create_CI_basis, represent_in_terms_of_ground, get_basis_shape
from memory_management import compute_truncated_array_length
from ci_dipole import (
    CI_exciton_small,
    CI_exciton_large,
    calculate_energies_and_oscillator_strengths,
)
from utils import save_data


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "example_dataset/Rx_3_Ry_3.5_Rz_2.25" 


def load_wavefunction(file_name, grid_spacing, material, band_label):
    """Load a wavefunction if a filename is given; otherwise return None."""
    if file_name is None:
        return None

    return Wavefunction.from_file(
        str(DATA_DIR) + "/",
        file_name,
        grid_spacing,
        material,
        band_label,
    )


def build_hamiltonian_dictionary(hams):
    """
    hams ordering:
    hams[0] = H_0
    hams[1] = J_LR
    hams[2] = K_SR
    hams[3] = K_LR0
    hams[4] = K_LR1
    hams[5] = K_LR2DD
    """
    H_0, J_LR, K_SR, K_LR0, K_LR1, K_LR2DD = hams

    return {
        "J_LR+K_SR": H_0 + J_LR + K_SR,
        "J_LR+K_LR0": H_0 + J_LR + K_LR0,
        "J_LR+K_LR1": H_0 + J_LR + K_LR1,
        "J_LR+K_LR2DD": H_0 + J_LR + K_LR2DD,
        "band_mixing": H_0 + J_LR + K_LR0 + K_LR1,
        "SR+DD": H_0 + J_LR + K_SR + K_LR2DD,
        "full": H_0 + J_LR + K_LR0 + K_LR1 + K_LR2DD,
    }


def print_component_diagnostics(hams):
    labels = {
        "J_LR": 1,
        "K_SR": 2,
        "K_LR0": 3,
        "K_LR1": 4,
        "K_LR2DD": 5,
    }

    for label, idx in labels.items():
        print(f"{label}:", repr(hams[idx]))
        print(f"{label}_eig:", LA.eig(hams[idx])[0])


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Input settings
    # -------------------------------------------------------------------------
    electron_file_name = "0.3307081259430eV.txt"
    hh_file_name = "0-1.261509141160eV_HH.txt"
    lh_file_name = "0-1.326217721890eV_LH.txt"

    grid_spacing = 0.5
    framework = "cv"
    memory_buffer = 0.5
    save_results = True

    material = Material(
        name="InAs",
        Ep=21.5,
        Eg=0.417,
        Delta_SO=0.39,
        Delta_SRK=0.19723521167947522,
        ep=15.15,
        a0=0.0529177210903,
    )

    # -------------------------------------------------------------------------
    # Load wavefunctions from example_dataset/
    # -------------------------------------------------------------------------
    E = load_wavefunction(electron_file_name, grid_spacing, material, "e")
    HH = load_wavefunction(hh_file_name, grid_spacing, material, "hh")
    LH = load_wavefunction(lh_file_name, grid_spacing, material, "lh")

    wavefunctions = (E, HH, LH)

    if E is None:
        raise ValueError("At least one electron wavefunction must be provided.")

    # -------------------------------------------------------------------------
    # Memory check and CI basis construction
    # -------------------------------------------------------------------------
    new_len, old_len = compute_truncated_array_length(
        E.data[:, 0],
        buffer_ratio=memory_buffer,
    )

    start = time.time()

    CI_basis, fock_space = create_CI_basis(wavefunctions, print_basis=True)
    _, _, _ = represent_in_terms_of_ground(CI_basis, fock_space)

    # -------------------------------------------------------------------------
    # Compute CI Hamiltonian components
    # -------------------------------------------------------------------------
    if old_len > new_len:
        number_chunks = old_len // new_len + 1

        for i in range(len(CI_basis)):
            CI_basis[i][0] = Wavefunction.split_wf(CI_basis[i][0], number_chunks)
            CI_basis[i][1] = Wavefunction.split_wf(CI_basis[i][1], number_chunks)

        hams = np.cdouble(
            CI_exciton_large(
                CI_basis,
                framework=framework,
                SR=True,
                BM0=True,
                BM1=True,
                DD=True,
            )
        )

        print_component_diagnostics(hams)

    else:
        hams = np.cdouble(
            CI_exciton_small(
                CI_basis,
                framework=framework,
                SR=True,
                BM0=True,
                BM1=True,
                DD=True,
            )
        )

    # Recreate unchunked basis for oscillator strengths
    CI_basis, fock_space = create_CI_basis(wavefunctions, print_basis=True)

    ham_dict = build_hamiltonian_dictionary(hams)

    e_os_dict = {}
    CIcs_dict = {}

    for key, H_CI in ham_dict.items():
        CIcs_dict[key], e_os_dict[key] = calculate_energies_and_oscillator_strengths(
            H_CI,
            CI_basis,
            fock_space,
        )

    print("(Energies, Oscillator strengths):", e_os_dict)
    print(time.time() - start, "seconds")

    # -------------------------------------------------------------------------
    # Save results to results/
    # -------------------------------------------------------------------------
    if save_results:
        save_data(
            E,
            hams,
            H_CI,
            e_os_dict,
            CIcs_dict,
            str(DATA_DIR),
            HH,
            LH,
        )


if __name__ == "__main__":
    main()

