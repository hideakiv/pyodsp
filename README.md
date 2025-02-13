# pyodsp
Pyomo interface for Decomposition of Structured Programs, inspired by [DSP](https://github.com/Argonne-National-Laboratory/DSP)

## Features
Pyodsp offers distributed algorithms for programming models written in [Pyomo](https://www.pyomo.org/).

### Benders Decomposition (bd)
- Decomposition based on complicating variables.
- Currently only supports linear programs (LP).
### Dual Decomposition (dd)
- Decomposition based on complicating constraints.
- Dual of Dantzig - Wolfe decomposition (DWD)
- Supports mixed-integer linear programs (MILP), but the solution is not guaranteed to be optimal.

## Prerequisites
### Solvers
- [HiGHS](https://highs.dev/)
Other solvers are not tested yet.
Additionally, the following is requrired for some of the algorithm.
- [Ipopt](https://github.com/coin-or/Ipopt)

### MPI
- [MPICH](https://www.mpich.org/)
OpenMPI is not tested.
