# Supported Solvers

`ruleopt` currently ships with solver backends for `HiGHS`, `Gurobi`, and `CPLEX`.

## HiGHS

`HiGHSSolver` is the default open-source solver in `ruleopt`.

Current defaults are tuned internally for faster large-sample RUG fits.
The current internal configuration uses:

```python
solver = "ipm"
run_crossover = "off"
presolve = "off"
threads = 0
ipm_optimality_tolerance = 1e-4
```

These settings usually reduce fit time, but they can change the selected rules and
the final model quality slightly compared with stricter solver settings.

## Gurobi

`GurobiSolver` uses the `gurobipy` interface and requires a valid Gurobi license.

## CPLEX

`CPLEXSolver` uses the `docplex` interface and requires a valid CPLEX installation
and license.
