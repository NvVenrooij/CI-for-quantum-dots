#%%
import numpy as np
from scipy.spatial.distance import cdist
from functools import cache


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
        """
        Extract the energy from filenames such as:

        0.2483927480530eV.txt
        0-1.190970497010eV.txt
        0-1.190970497010eV_LH.txt
        0-1.190970497010eV_HH.txt
        """

        energy_string = wf_file

        # Remove band labels if present
        energy_string = energy_string.replace("_LH", "")
        energy_string = energy_string.replace("_HH", "")

        # Remove file extension / unit
        energy_string = energy_string.replace("eV.txt", "")

        # Old convention: negative energies are written as 0-1.23 instead of -1.23
        if energy_string.startswith("0-"):
            return -float(energy_string.replace("0-", ""))

        return float(energy_string)
    
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

