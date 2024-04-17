from libc.stdlib cimport free, realloc
from cython cimport boundscheck, wraparound

import numpy as np
from numpy import float64 as DOUBLE

cdef class Coefficients:
    """
    Represents coefficients for a model, including the values, rows, columns, and costs.

    Attributes:
    -----------
    yvals : cnp.ndarray[DOUBLE_t, ndim=1]
        The coefficient values.
    rows : cnp.ndarray[SIZE_t, ndim=1]
        Row indices for the coefficients.
    cols : cnp.ndarray[SIZE_t, ndim=1]
        Column indices for the coefficients.
    costs : cnp.ndarray[DOUBLE_t, ndim=1]
        The cost associated with each coefficient.
    """
    def __init__(self, DOUBLE_t[:] yvals = np.empty(shape=(0), dtype=DOUBLE),
                 SIZE_t[:] rows = np.empty(shape=(0), dtype=np.intp),
                 SIZE_t[:] cols = np.empty(shape=(0), dtype=np.intp),
                 DOUBLE_t[:] costs = np.empty(shape=(0), dtype=DOUBLE)):
        self.yvals = yvals
        self.rows = rows
        self.cols = cols
        self.costs = costs

    def cleanup(self):
        self.yvals = np.empty(shape=(0), dtype=DOUBLE)
        self.rows = np.empty(shape=(0), dtype=np.intp)
        self.cols = np.empty(shape=(0), dtype=np.intp)
        self.costs = np.empty(shape=(0), dtype=DOUBLE)

cdef struct ClauseStruct:
    # A structure representing a clause in a rule, including the feature index, 
    # upper and lower bounds, and a flag for handling NaN values.
    int feature
    double ub
    double lb
    bint na

cdef class Rule:
    def __cinit__(self):
        """
        Represents a rule consisting of multiple clauses used to make decisions
        based on feature values.

        Attributes:
        -----------
        n_clauses : int
            The number of clauses in the rule.
        clauses : ClauseStruct*
            Pointer to the array of ClauseStruct representing the clauses in the rule.
        label : Any
            The label associated with the rule.
        weight : float
            The weight of the rule.
        sdist : Any
            The distribution of scores associated with the rule.
        """

        self.n_clauses = 0
        self.label = None 
        self.weight = None
        self.sdist = None

    def __len__(self):
        return self.n_clauses

    def __dealloc__(self):
        if self.n_clauses > 0:
            free(self.clauses)

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False

        if self.n_clauses != other.n_clauses:
            return False

        self_clauses = {self._get_clause(i) for i in range(self.n_clauses)}
        other_clauses = {other._get_clause(i) for i in range(other.n_clauses)}

        return self_clauses == other_clauses


    @boundscheck(False) 
    @wraparound(False)  
    cpdef tuple _get_clause(self, int index):
        cdef int feature = self.clauses[index].feature
        cdef double ub = self.clauses[index].ub
        cdef double lb = self.clauses[index].lb
        cdef bint na = self.clauses[index].na

        return feature, ub, lb, na


    @boundscheck(False) 
    @wraparound(False)  
    cpdef add_clause(self, int feature, double ub, double lb, bint na):
        """
        Adds a new clause to the rule or updates an existing clause for the given feature.

        Parameters:
        -----------
        feature : int
            The feature index to which the clause applies.
        ub : double
            The upper bound for the feature value.
        lb : double
            The lower bound for the feature value.
        na : bint
            A flag indicating whether NaN values should be considered as matching this clause.
        """
        cdef bint check_exist = 0
        cdef int i


        for i in range(self.n_clauses):
            if feature == self.clauses[i].feature:
                self.clauses[i].lb = max(self.clauses[i].lb, lb)
                self.clauses[i].ub = min(self.clauses[i].ub, ub)
                check_exist = 1
                break

        if check_exist == 0:
            self.clauses = <ClauseStruct*>realloc(self.clauses, (self.n_clauses + 1) * sizeof(ClauseStruct))

            self.clauses[self.n_clauses].feature = feature
            self.clauses[self.n_clauses].ub = ub
            self.clauses[self.n_clauses].lb = lb
            self.clauses[self.n_clauses].na = na
            
            self.n_clauses += 1

    @boundscheck(False)
    @wraparound(False) 
    cdef char[:] _check_rule_nogil(self, float[:,:] X) noexcept:
        """
        Checks if the rule applies to the given feature values, 
        without Python GIL and exception handling.

        Parameters:
        -----------
        X : cnp.ndarray[cnp.float32_t, ndim=2]
            The 2D array of feature values to check against the rule. Each row represents a different set of features.

        Returns:
        --------
        cnp.ndarray[cnp.uint8_t, ndim=1]
            An array where each element corresponds to a row in X, indicating True (1) if the rule applies, False (0) otherwise.
        """
        cdef int i, j
        cdef char[:] result_data = np.ones(X.shape[0], dtype=np.int8)

        with nogil:
            for i in range(X.shape[0]):
                for j in range(self.n_clauses):
                    if not self._check_clause_nogil(X[i], j):
                        result_data[i] = 0  # Rule doesn't apply
                        break

        return result_data

    @boundscheck(False) 
    @wraparound(False)
    cdef bint _check_clause_nogil(self, float[:] X, int idx) noexcept nogil:
        """
        Checks if a specific clause of the rule applies to the given feature values, 
        without Python GIL and exception handling.

        Parameters:
        -----------
        X : cnp.ndarray[cnp.float32_t, ndim=1]
            The array of feature values to check against the clause.
        idx : int
            The index of the clause to check.

        Returns:
        --------
        bint
            True if the clause applies, False otherwise.
        """
        cdef ClauseStruct clause = self.clauses[idx]
        cdef float val = X[clause.feature]

        if val != val:  # Checking for NaN
            return clause.na
        return clause.lb < val <= clause.ub


    cpdef cnp.ndarray[cnp.uint8_t, ndim=1] check_rule(self, float[:, :] X):
        """
        Checks if the rule applies to the given feature values.

        Parameters:
        -----------
        X : cnp.ndarray[cnp.float32_t, ndim=2]
            The 2D array of feature values to check against the rule. Each row represents a different set of features.

        Returns:
        --------
        np.ndarray[cnp.uint8_t, ndim=1]
            An array indicating whether the rule applies to each feature value. Each element in the array corresponds to a row in X, 
            indicating True (1) if the rule applies, False (0) otherwise.
        """
        cdef char[:] result = self._check_rule_nogil(X)
        return np.asarray(result, dtype=np.uint8)


    def to_text(self, feature_names=None):
        """
        Converts the rule to a human-readable text representation, using optional 
        feature names for clarity.

        Parameters:
        ----------
        feature_names : list, optional
            A list of feature names corresponding to feature indices. If None, f
            eature indices are used.

        Returns:
        -------
        str
            A human-readable string representation of the rule.
        """
        print_text = ""
        for i in range(self.n_clauses):
            feature_label = (f"x[{self.clauses[i].feature}]"
                             if feature_names is None
                             else feature_names[self.clauses[i].feature])
            na_string = " or null" if self.clauses[i].na else " and not null"
            print_text += f"{self.clauses[i].lb:<9.2f} < {feature_label:<9} <= {self.clauses[i].ub:<9.2f}{na_string}\n"
        return print_text.rstrip("\n")

    def to_dict(self, feature_names=None):
        """
        Converts the rule to a dictionary representation, using optional feature names for keys.

        Parameters:
        ----------
        feature_names : list, optional
            A list of feature names corresponding to feature indices. If None, feature indices are used as keys.

        Returns:
        -------
        dict
            A dictionary representation of the rule, with feature names or indices as keys and clause details as values.
        """
        return {
            (feature_names[self.clauses[i].feature] if feature_names else self.clauses[i].feature): 
            {"lb": self.clauses[i].lb, "ub": self.clauses[i].ub, "na": self.clauses[i].na}
            for i in range(self.n_clauses)
        }