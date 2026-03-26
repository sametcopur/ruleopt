from libc.stdlib cimport free, malloc, realloc
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

cdef struct ObliqueClauseStruct:
    double* weights
    int* features
    double threshold
    int n_terms
    bint is_left

cdef class Rule:
    def __cinit__(self):
        self.n_clauses = 0
        self.n_oblique_clauses = 0
        self.label = None
        self.weight = None
        self.sdist = None

    def __len__(self):
        return self.n_clauses + self.n_oblique_clauses

    def __dealloc__(self):
        cdef int i
        if self.n_clauses > 0:
            free(self.clauses)
        if self.n_oblique_clauses > 0:
            for i in range(self.n_oblique_clauses):
                free(self.oblique_clauses[i].weights)
                free(self.oblique_clauses[i].features)
            free(self.oblique_clauses)

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False

        if self.n_clauses != other.n_clauses:
            return False
        if self.n_oblique_clauses != other.n_oblique_clauses:
            return False

        self_clauses = {self._get_clause(i) for i in range(self.n_clauses)}
        other_clauses = {other._get_clause(i) for i in range(other.n_clauses)}
        if self_clauses != other_clauses:
            return False

        self_oblique = {self._get_oblique_clause(i) for i in range(self.n_oblique_clauses)}
        other_oblique = {other._get_oblique_clause(i) for i in range(other.n_oblique_clauses)}
        return self_oblique == other_oblique

    # ── Single-feature clause methods (unchanged) ─────────────────

    cpdef tuple _get_clause(self, int index):
        cdef int feature = self.clauses[index].feature
        cdef double ub = self.clauses[index].ub
        cdef double lb = self.clauses[index].lb
        cdef bint na = self.clauses[index].na

        return feature, ub, lb, na

    cpdef add_clause(self, int feature, double ub, double lb, bint na):
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

    cdef bint _check_clause_nogil(self, float[:] X, int idx) noexcept nogil:
        cdef ClauseStruct clause = self.clauses[idx]
        cdef float val = X[clause.feature]

        if val != val:  # NaN
            return clause.na
        return clause.lb < val <= clause.ub

    # ── Oblique clause methods (new) ──────────────────────────────

    cpdef tuple _get_oblique_clause(self, int index):
        cdef ObliqueClauseStruct oc = self.oblique_clauses[index]
        cdef list w = [oc.weights[i] for i in range(oc.n_terms)]
        cdef list f = [oc.features[i] for i in range(oc.n_terms)]
        return (tuple(w), tuple(f), oc.threshold, oc.is_left)

    cpdef add_oblique_clause(self, list weights, list features, double threshold, bint is_left):
        cdef int n = len(weights)
        cdef int i

        self.oblique_clauses = <ObliqueClauseStruct*>realloc(
            self.oblique_clauses,
            (self.n_oblique_clauses + 1) * sizeof(ObliqueClauseStruct)
        )

        cdef ObliqueClauseStruct* oc = &self.oblique_clauses[self.n_oblique_clauses]
        oc.n_terms = n
        oc.threshold = threshold
        oc.is_left = is_left
        oc.weights = <double*>malloc(n * sizeof(double))
        oc.features = <int*>malloc(n * sizeof(int))

        for i in range(n):
            oc.weights[i] = weights[i]
            oc.features[i] = features[i]

        self.n_oblique_clauses += 1

    cdef bint _check_oblique_clause_nogil(self, float[:] X, int idx) noexcept nogil:
        cdef ObliqueClauseStruct* oc = &self.oblique_clauses[idx]
        cdef double dot = 0.0
        cdef float val
        cdef int i

        for i in range(oc.n_terms):
            val = X[oc.features[i]]
            if val != val:  # NaN
                return 0
            dot += oc.weights[i] * val

        if oc.is_left:
            return dot < oc.threshold
        else:
            return dot >= oc.threshold

    # ── Rule evaluation ───────────────────────────────────────────

    cdef char[:] _check_rule_nogil(self, float[:,:] X) noexcept:
        cdef int i, j, k
        cdef bint passed
        cdef char[:] result_data = np.ones(X.shape[0], dtype=np.int8)
        cdef int n_samples = X.shape[0]
        cdef int n_clauses = self.n_clauses
        cdef int n_oblique = self.n_oblique_clauses
        cdef ClauseStruct* clauses = self.clauses
        cdef ObliqueClauseStruct* oblique = self.oblique_clauses
        cdef float val
        cdef double dot

        with nogil:
            for i in range(n_samples):
                passed = 1
                for j in range(n_clauses):
                    val = X[i, clauses[j].feature]
                    if val != val:
                        if not clauses[j].na:
                            passed = 0
                            break
                    elif not (clauses[j].lb < val <= clauses[j].ub):
                        passed = 0
                        break
                if passed and n_oblique > 0:
                    for j in range(n_oblique):
                        dot = 0.0
                        for k in range(oblique[j].n_terms):
                            val = X[i, oblique[j].features[k]]
                            if val != val:
                                passed = 0
                                break
                            dot = dot + oblique[j].weights[k] * val
                        if not passed:
                            break
                        if oblique[j].is_left:
                            if not (dot < oblique[j].threshold):
                                passed = 0
                                break
                        else:
                            if not (dot >= oblique[j].threshold):
                                passed = 0
                                break
                if not passed:
                    result_data[i] = 0

        return result_data

    cpdef cnp.ndarray[cnp.uint8_t, ndim=1] check_rule(self, float[:, :] X):
        cdef char[:] result = self._check_rule_nogil(X)
        return np.asarray(result).view(np.uint8)

    # ── Display ───────────────────────────────────────────────────

    def to_text(self, feature_names=None):
        print_text = ""
        for i in range(self.n_clauses):
            feature_label = (f"x[{self.clauses[i].feature}]"
                             if feature_names is None
                             else feature_names[self.clauses[i].feature])
            na_string = " or null" if self.clauses[i].na else " and not null"
            print_text += f"{self.clauses[i].lb:<9.2f} < {feature_label:<9} <= {self.clauses[i].ub:<9.2f}{na_string}\n"

        for i in range(self.n_oblique_clauses):
            oc = self.oblique_clauses[i]
            terms = []
            for j in range(oc.n_terms):
                fname = (f"x[{oc.features[j]}]"
                         if feature_names is None
                         else feature_names[oc.features[j]])
                terms.append(f"{oc.weights[j]:.4f}*{fname}")
            expr = " + ".join(terms)
            op = "<" if oc.is_left else ">="
            print_text += f"{expr} {op} {oc.threshold:.4f}\n"

        return print_text.rstrip("\n")

    def to_dict(self, feature_names=None):
        result = {}
        for i in range(self.n_clauses):
            key = (feature_names[self.clauses[i].feature] if feature_names else self.clauses[i].feature)
            result[key] = {
                "lb": self.clauses[i].lb, "ub": self.clauses[i].ub, "na": self.clauses[i].na
            }

        for i in range(self.n_oblique_clauses):
            oc = self.oblique_clauses[i]
            features_key = tuple(
                feature_names[oc.features[j]] if feature_names else oc.features[j]
                for j in range(oc.n_terms)
            )
            weights_list = [oc.weights[j] for j in range(oc.n_terms)]
            result[features_key] = {
                "type": "oblique",
                "weights": weights_list,
                "threshold": oc.threshold,
                "direction": "<" if oc.is_left else ">=",
            }

        return result
