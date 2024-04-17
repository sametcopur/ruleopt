cimport numpy as cnp

ctypedef cnp.npy_float64 DOUBLE_t
ctypedef cnp.npy_float32 DTYPE_t
ctypedef cnp.npy_intp SIZE_t

cdef class Coefficients:
    cdef public DOUBLE_t[:] yvals
    cdef public SIZE_t[:] rows
    cdef public SIZE_t[:] cols
    cdef public DOUBLE_t[:] costs

cdef struct ClauseStruct:
    int feature
    double ub
    double lb
    bint na

cdef class Rule:
    cdef ClauseStruct* clauses
    cdef public int n_clauses
    cdef public object label
    cdef public object weight
    cdef public object sdist
    cpdef add_clause(self, int feature, double ub, double lb, bint na)
    cpdef tuple _get_clause(self, int index)
    cdef char[:] _check_rule_nogil(self, float[:,:] X) noexcept 
    cdef bint _check_clause_nogil(self, float[:] X, int idx) noexcept nogil 
    cpdef cnp.ndarray[cnp.uint8_t, ndim=1] check_rule(self, float[:,:] X)
