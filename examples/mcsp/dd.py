from pathlib import Path
import pyomo.environ as pyo

from mcsp import master_problem, sub_problem

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.dd.alg_root_bm import DdAlgRootBm
from pyodsp.dec.dd.alg_root_pbm import DdAlgRootPbm
from pyodsp.dec.dd.alg_leaf import DdAlgLeaf
from pyodsp.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodsp.dec.dd.run import DdRun
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig


def main(
    N: list[int],
    P: int,
    d: list[int],
    L: list[int],
    c: list[float],
    l: list[int],
    solver="appsi_highs",
):
    nodes = []
    master = create_master(N, P, d, solver, pbm=True)

    nodes.append(master)

    K = len(N)

    for k in range(K):
        sub = create_sub(k, N[k], P, L[k], c[k], l, solver)
        master.add_child(k + 1)
        nodes.append(sub)

    dd_run = DdRun(nodes, Path("output/mcsp/dd"))
    dd_run.run()


def create_master(
    N: list[int],
    P: int,
    d: list[int],
    solver="appsi_highs",
    pbm=False,
) -> DecNodeRoot:
    m = pyo.ConcreteModel()
    master_problem(m, N, P, d)

    K = len(N)
    vars_dn = {}
    for k in range(K):
        xs = []
        for p in range(P):
            xs.append(m.xtot[k, p])
        vars_dn[k + 1] = xs

    final_config = SolverConfig(solver_name=solver, kwargs={"tee": True})
    if pbm:
        alg_config = SolverConfig(solver_name="ipopt")
        root_alg = DdAlgRootPbm(m, True, alg_config, final_config, vars_dn)
    else:
        alg_config = SolverConfig(solver_name=solver)
        root_alg = DdAlgRootBm(m, True, alg_config, final_config, vars_dn)
    root_node = DecNodeRoot(0, root_alg)
    return root_node


def create_sub(
    k: int, N: int, P: int, L: int, c: float, l: list[int], solver="appsi_highs"
) -> DecNodeLeaf:
    m = pyo.ConcreteModel()
    sub_problem(m, N, P, L, c, l)
    m.obj = pyo.Objective(expr=m.objexpr, sense=pyo.minimize)

    vars_up = []
    for p in range(P):
        vars_up.append(m.xtot[p])

    config = SolverConfig(solver_name=solver)
    sub_solver = PyomoSolver(m, config, vars_up)
    sub_alg = DdAlgLeafPyomo(sub_solver)
    leaf_node = DecNodeLeaf(k + 1, sub_alg)
    leaf_node.add_parent(0)

    return leaf_node


if __name__ == "__main__":
    N = [2000]
    P = 5
    d = [205, 2321, 143, 1089, 117]
    L = [110]
    c = [1.0]
    l = [70, 40, 55, 25, 35]
    main(N, P, d, L, c, l)
