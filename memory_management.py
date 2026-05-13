
import numpy as np
from scipy.spatial.distance import cdist
import psutil
import math


def compute_truncated_array_length(arr, buffer_ratio=0.5):
    """
    Compute the maximum length of a truncated array based on available memory and buffer ratio.

    Parameters:
    - arr1: First array.
    - arr2: Second array.
    - max_memory: Maximum available memory (default: None).
    - buffer_ratio: Ratio of allocated memory to maximum memory (default: 0.8).

    Returns:
    - max_array_length: Maximum length of the truncated array.

    """

    # Compute maximum amount of memory available
    # Depending on the buffer ratio, the allocated memory <= max memory
   
    max_memory = psutil.virtual_memory().available
    available_memory = int(max_memory * buffer_ratio)

    # Determine the size that the arrays will be
    old_array_length = arr.shape[0]
    required_memory_outer_product_array = arr.itemsize * arr.size * arr.size
    required_memory_dr_vec_array = required_memory_outer_product_array*3/2 #division by 2 because it is not a np.clongdouble 128 but float 64 
    required_memory_dist_array = required_memory_dr_vec_array*0.5

    # Print the computed values for analysis
    print("Maximum available memory =", max_memory/ (1024 ** 3), "GB")
    print("Available memory (including buffer) =", available_memory/ (1024 ** 3), "GB")
    print("Required memory for outer product wf array =", required_memory_outer_product_array/ (1024 ** 3), "GB")
    print("Required memory for dr_vec array =", required_memory_dr_vec_array/ (1024 ** 3), "GB")
    print("Required memory for dist array =", required_memory_dist_array/ (1024 ** 3), "GB")

    required_memory = required_memory_outer_product_array + required_memory_dr_vec_array + required_memory_dist_array
    print("Required Memory: ", required_memory/ (1024 ** 3), "GB")


    # Determine the maximum chunk size by taking the minimum value between
    # available_memory and the memory required for the 2D array
    if available_memory <= required_memory:
        max_arr_size = int(available_memory*(required_memory_dist_array/(required_memory_outer_product_array+required_memory_dr_vec_array+required_memory_dist_array))) #the arrays are all the same size but 
        print("max arr size =", max_arr_size)
        max_number_elements = int(max_arr_size / arr.itemsize)
        print("Not enough memory")
        print("Max array memory =", max_arr_size/ (1024 ** 3), "GB")
        print("Max number of elements per chunk =", max_number_elements)

        # Calculate the maximum array length based on the maximum number of elements per chunk
        print("Old array length =", old_array_length)
        new_array_length = math.isqrt(max_number_elements)
        print("New array length =", new_array_length)
        #max_array_length = int(integer_square_mne / 8)  # divided by 8 because array is l x 8

        return new_array_length, old_array_length  
    
    else:
        max_arr_size = required_memory_outer_product_array
        max_number_elements = int(arr.size * arr.size)
        new_array_length = old_array_length
        print("Enough spare memory")
        print("Old array length =", old_array_length)
        print("New array length =", new_array_length)

        return new_array_length, old_array_length





def slice_arrays(arr, coo, trunc_array_length):
    """
    Slice arrays into smaller chunks based on the specified truncation array length.

    Parameters:
    - WF: Array to be sliced.
    - coo: Coordinate array to be sliced.
    - trunc_array_length: Length of the truncated arrays.

    Returns:
    - WF_arrays: list containing sliced arrays.
    - coos: Sliced arrays of coo.

    """

    # Compute the quotient and remainder of WF.size divided by trunc_array_length
    quotient, remainder = divmod(arr.size, trunc_array_length)

    # Compute the split points to split WF and coo arrays into smaller chunks
    #split_points = [i * trunc_array_length for i in np.arange(1, quotient, 1)]

    # Split WF array into smaller chunks based on the split points
    #WF_arrays = np.split(WF, split_points, axis=0)
    arrays = np.array_split(arr, len(arr)// trunc_array_length, axis=0)

    # Flatten WF array
    #for i in range(quotient):
    #    WF_arrays[i] = WF_arrays[i].flatten()

    # Split coo array into smaller chunks based on the split points
    coos = np.array_split(coo, len(coo)//trunc_array_length, axis=0)

    return arrays, coos
