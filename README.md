# Configuration Interaction for Quantum Dots

Calculates excitonic eigenenergies and fine structure splitting for quantum
dots (QDs) using configuration interaction (CI) methods. This computational
framework takes electron and hole single-particle eigenstates as input and
computes the many-body exciton states including Coulomb interactions and
exchange effects. The single particle eigenstates can be obtained from
8-band k·p envelope simulations.

Quantum dots are nanoscale semiconductor devices with applications spanning
displays to quantum information technology. The optical properties of these
devices depend on quasiparticles called excitons — bound states formed by
coupling between single-particle electron and hole wavefunctions. The
quantum dot geometry dictates the wavefunction shapes and thus affects
exciton eigenenergies. Understanding the exciton energy spectrum, or fine
structure, is crucial for implementing quantum dots in future quantum
technologies.

## Features

- Configuration interaction calculations for exciton states
- Support for various exchange interaction terms (short-range, long-range,
  band mixing)
- Memory-optimized calculations for large quantum dot systems via
  automatic wavefunction chunking
- Oscillator strength calculations for optical transitions and exciton
  classification
- Extended-precision (`np.clongdouble`) matrix assembly with safe
  downcasting for `numpy.linalg` compatibility

## Applications

- Fine structure splitting analysis in semiconductor quantum dots
- Excitonic properties of InAs/InP droplet epitaxy quantum dots
- Investigation of symmetry breaking effects on exciton states

## Installation

```bash
git clone https://github.com/NvVenrooij/CI-for-quantum-dots.git
cd CI-for-quantum-dots
pip install -r requirements.txt
```

Requires Python 3.9 or newer.

## Quick start

The repository ships with an example dataset for a single quantum dot
geometry (`example_dataset/Rx_3_Ry_3.5_Rz_2.25/`) containing the lowest
electron, heavy-hole and light-hole wavefunctions. To run a CI
calculation on it:

```bash
python run_ci.py
```

This will:

1. Load the three wavefunctions from `example_dataset/`.
2. Check available memory and decide whether to chunk the wavefunctions.
3. Build the CI Hamiltonian components (`H_0`, `J_LR`, `K_SR`, `K_LR_0`,
   `K_LR_1`, `K_LR_2_DD`).
4. Combine them into several physically interesting Hamiltonians
   (`full`, `band_mixing`, `SR+DD`, …), diagonalise each, and compute
   oscillator strengths.
5. Pickle all results to `example_dataset/.../CI_calculation_e_hh_lh.pkl`.

To run on your own data, replace the contents of `example_dataset/` (or
point `DATA_DIR` in `run_ci.py` somewhere else) and update the filenames
and material parameters at the top of `main()`.

## Input format

Each wavefunction file is a plain-text export from the upstream 8-band
k·p solver. The expected layout is:

- 5 header lines (skipped).
- One row per grid point with 3 real-valued coordinates followed by 8
  complex amplitudes (one per band), written as `(real, imag)` pairs.

Filenames encode the eigenenergy in eV and the band:

```text
0.3307081259430eV.txt        # electron,    +0.331 eV
0-1.261509141160eV_HH.txt    # heavy hole,  -1.262 eV
0-1.326217721890eV_LH.txt    # light hole,  -1.326 eV
```

## Input wavefunctions

The single-particle electron and hole wavefunctions used as input are
computed with the strain-dependent 8-band k·p envelope-function method
of Pryor [1], in the basis ordering of Bahder [2]. That solver is not
publicly available, so a small example dataset is provided in
`example_dataset/` to make the CI calculation runnable out of the box.

If you wish to use wavefunctions from another envelope-function code
(e.g. one of the open-source k·p packages now available), note that the
Bloch-band ordering assumed throughout `interactions.py` and
`material.py` is specific to the Bahder basis. Inputs in a different
basis must be transformed accordingly before they can be used here.

**References**

1. C. Pryor, *Eight-band calculations of strained InAs/GaAs quantum dots
   compared with one-, four-, and six-band approximations*,
   Phys. Rev. B **57**, 7190 (1998).
   [doi:10.1103/PhysRevB.57.7190](https://doi.org/10.1103/PhysRevB.57.7190)
2. T. B. Bahder, *Eight-band k·p model of strained zinc-blende crystals*,
   Phys. Rev. B **41**, 11992 (1990).
   [doi:10.1103/PhysRevB.41.11992](https://doi.org/10.1103/PhysRevB.41.11992)

## Module layout

| File | Purpose |
|---|---|
| `material.py` | `Material` class and 8-band dipole matrices |
| `wavefunction.py` | `Wavefunction` class: file I/O, time reversal, chunking |
| `basis.py` | CI basis construction and Fock-space mapping |
| `interactions.py` | Geometry, matrix elements, J/K interaction kernels |
| `memory_management.py` | Memory budgeting and array slicing helpers |
| `ci_dipole.py` | CI Hamiltonian assembly and oscillator strengths |
| `utils.py` | Result serialisation |
| `run_ci.py` | End-to-end example entry point |

## Output

Results are written as a single pickle (`CI_calculation_<bands>.pkl`)
containing the input single-particle energies, the individual interaction
Hamiltonians, the assembled CI Hamiltonian, all CI coefficients, and the
list of `(eigenenergy, oscillator_strength)` tuples for each physical
choice of interaction set.

## Related publication

This code was used in:

> N. R. S. van Venrooij, A. R. da Cruz, R. S. R. Gajjella, P. M. Koenraad,
> C. E. Pryor, and M. E. Flatté, *Fine structure splitting cancellation
> in highly asymmetric InAs/InP droplet epitaxy quantum dots*,
> Phys. Rev. B **109**, L201405 (2024).
> [doi:10.1103/PhysRevB.109.L201405](https://doi.org/10.1103/PhysRevB.109.L201405)

If you use this code in academic work, please cite the paper above.

## Requirements

- Python ≥ 3.9
- NumPy, SciPy, Numba, psutil (see `requirements.txt`)
- Input: single-particle electron/hole wavefunctions from 8-band k·p
  envelope-function calculations
- Output: excitonic eigenenergies, oscillator strengths, CI coefficients

## License
MIT 