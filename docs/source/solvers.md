# Supported Solvers

`ruleopt` currently ships with solver backends for `HiGHS`, `Gurobi`, and `CPLEX`.

## HiGHS

`HiGHSSolver` is the default open-source solver in `ruleopt`.

`GurobiSolver` uses the `gurobipy` interface and requires a valid Gurobi license.

`CPLEXSolver` uses the `docplex` interface and requires a valid CPLEX installation
and license.
