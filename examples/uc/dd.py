from pathlib import Path
import pyomo.environ as pyo

from uc import balance, single_generator
from heuristics import UcHeuristicRoot
from params import UcParams, create_random

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.dd.alg_root_bm import DdAlgRootBm
from pyodsp.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodsp.dec.dd.run import DdRun
from pyodsp.dec.dd.mip_heuristic_root import MipHeuristicRoot
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig


def main(
    num_time: int,
    num_gens: int,
    demand: list[float],
    params: dict[int, UcParams],
    solver="appsi_highs",
):
    nodes = []
    master = create_master(num_time, num_gens, demand, params, solver, pbm=True)

    nodes.append(master)

    for k in range(1, num_gens + 1):
        sub = create_sub(k, num_time, params, solver)
        master.add_child(k)
        nodes.append(sub)

    dd_run = DdRun(nodes, Path("output/uc/dd"))
    dd_run.run()


def create_master(
    num_time: int,
    num_gens: int,
    demand: list[float],
    params: dict[int, UcParams],
    solver="appsi_highs",
    pbm=False,
) -> DecNodeRoot:
    model = pyo.ConcreteModel()

    balance(model, num_time, num_gens, demand)

    vars_dn = {}
    for k in range(1, num_gens + 1):
        xs = []
        for t in range(1, num_time + 1):
            xs.append(model.p[k, t])
        vars_dn[k] = xs

    final_config = SolverConfig(solver_name=solver, kwargs={"tee": True})
    heuristics = UcHeuristicRoot(final_config, params, num_time)

    if pbm:
        alg_config = SolverConfig(solver_name="ipopt")
        root_alg = DdAlgRootBm(
            model, True, alg_config, vars_dn, heuristics, mode="proximal"
        )
    else:
        alg_config = SolverConfig(solver_name=solver)
        root_alg = DdAlgRootBm(model, True, alg_config, vars_dn, heuristics)
    root_node = DecNodeRoot(0, root_alg)
    return root_node


def create_sub(
    k: int, num_time: int, params: dict[int, UcParams], solver="appsi_highs"
) -> DecNodeLeaf:
    model = pyo.ConcreteModel()
    single_generator(model, num_time, params[k])
    model.obj = pyo.Objective(expr=model.objexpr, sense=pyo.minimize)

    vars_up = []
    for t in range(1, num_time + 1):
        vars_up.append(model.p[t])

    config = SolverConfig(solver_name=solver)
    sub_solver = PyomoSolver(model, config, vars_up)
    sub_alg = DdAlgLeafPyomo(sub_solver)
    leaf_node = DecNodeLeaf(k, sub_alg)
    leaf_node.add_parent(0)

    return leaf_node


if __name__ == "__main__":
    num_day = 1
    num_gens = 20
    num_seg = 5
    num_time, demand, params = create_random(num_day, num_gens, num_seg)
    main(num_time, num_gens, demand, params)
