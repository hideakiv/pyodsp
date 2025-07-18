from pathlib import Path

import pyomo.environ as pyo

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.dd.alg_root_bm import DdAlgRootBm
from pyodsp.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodsp.dec.dd.run import DdRun

from utils import get_args, assert_approximately_equal


def create_master(solver="appsi_highs") -> DecNodeRoot:
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)
    block.y1 = pyo.Var(within=pyo.Reals)
    block.y2 = pyo.Var(within=pyo.Reals)
    vars_dn = {1: [block.x1, block.x2], 2: [block.y1, block.y2]}

    block.c1 = pyo.Constraint(
        expr=-1 * block.x1 + 5 * block.x2 + 7 * block.y1 - 6 * block.y2 == 1
    )

    alg_config = SolverConfig(solver_name=solver)
    root_alg = DdAlgRootBm(block, True, alg_config, vars_dn)
    root_node = DecNodeRoot(0, root_alg)
    return root_node


cost = {1: [1, 2], 2: [3, 4]}


def create_sub(i, solver="appsi_highs") -> DecNodeLeaf:
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)
    vars_up = [block.x1, block.x2]

    block.obj = pyo.Objective(
        expr=cost[i][0] * block.x1 + cost[i][1] * block.x2, sense=pyo.minimize
    )
    if i == 1:
        block.c1 = pyo.Constraint(expr=block.x1 + block.x2 >= 1)
        block.c2 = pyo.Constraint(expr=-block.x1 + 2 * block.x2 >= 0)
        block.c3 = pyo.Constraint(expr=block.x2 >= 1)
    elif i == 2:
        block.c1 = pyo.Constraint(expr=3 * block.x1 - block.x2 >= 2)
        block.c2 = pyo.Constraint(expr=block.x1 + 2 * block.x2 >= 3)

    config = SolverConfig(solver_name=solver)
    sub_solver = PyomoSolver(block, config, vars_up)
    sub_alg = DdAlgLeafPyomo(sub_solver)
    leaf_node = DecNodeLeaf(i, sub_alg)
    leaf_node.add_parent(0)
    return leaf_node


def main():
    args = get_args()

    master = create_master(args.solver)
    sub_1 = create_sub(1, args.solver)
    sub_2 = create_sub(2, args.solver)

    master.add_child(1)
    master.add_child(2)

    dd_run = DdRun([master, sub_1, sub_2], Path("output/dd/ray"))
    dd_run.run()

    assert_approximately_equal(master.alg_root.bm.obj_bound[-1], 15.09090909090909)


if __name__ == "__main__":
    main()
