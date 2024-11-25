import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound


@boundscheck(False) 
@wraparound(False)  
cpdef cnp.ndarray[cnp.float64_t, ndim=1] fill_betas(
    int n,
    double[::1] duals_unique,
    long[::1] inverse_indices,
    float[::1] sample_weight,
    object rng  # Random number generator, keep it generic
):
    cdef:
        double[:] betas = np.zeros(n, dtype=np.float64)
        int[::1] indexes = rng.permutation(np.arange(n, dtype=np.int32))
        float tol = 1e-4  # Precision threshold
        Py_ssize_t i, value_index
        double value, remaining_dual
        float weight

    with nogil:
        for i in indexes:
            value_index = inverse_indices[i]
            remaining_dual = duals_unique[value_index]
            
            if remaining_dual > tol:  # Only process if meaningful dual value remains
                weight = sample_weight[i]
                
                if remaining_dual >= weight:
                    # If enough dual value remains, assign full weight
                    betas[i] = weight
                    duals_unique[value_index] -= weight
                else:
                    # If not enough dual value remains, assign what's left
                    betas[i] = remaining_dual
                    duals_unique[value_index] = 0.0
                    
    return np.asarray(betas)

        