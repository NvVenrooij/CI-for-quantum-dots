# Configuration Interaction for Quantum Dots

Calculates excitonic eigenenergies and fine structure splitting for quantum dots (QDs) using configuration interaction (CI) methods. This computational framework takes electron and hole single-particle eigenstates as input and computes the many-body exciton states including Coulomb interactions and exchange effects. The single particle eigenstates can be obtained from 8-band k·p envelope simulations.

Quantum dots are nanoscale semiconductor devices with applications spanning displays to quantum information technology. The optical properties of these devices depend on quasiparticles called excitons - bound states formed by coupling between single-particle electron and hole wavefunctions. The quantum dot geometry dictates the wavefunction shapes and thus affects exciton eigenenergies. Understanding the exciton energy spectrum, or fine structure, is crucial for implementing quantum dots in future quantum technologies.

## Features
- Configuration interaction calculations for exciton states
- Support for various exchange interaction terms (short-range, long-range, band mixing)
- Memory-optimized calculations for large quantum dot systems
- Oscillator strength calculations for optical transitions and exciton classification

## Applications
- Fine structure splitting analysis in semiconductor quantum dots
- Excitonic properties of InAs/InP droplet epitaxy quantum dots
- Investigation of symmetry breaking effects on exciton states

## Related Publication
This code was used in the research published in:  https://doi.org/10.1103/PhysRevB.109.L201405

## Requirements
- Python with NumPy, SciPy, Numba
- Input: Single-particle electron/hole wavefunctions from quantum dot calculations
- Output: Excitonic eigenenergies, oscillator strengths, CI coefficients
