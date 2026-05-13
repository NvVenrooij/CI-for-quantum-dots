
import numpy as np

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
