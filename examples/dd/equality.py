from pathlib import Path

import pyomo.environ as pyo

from pyodsp.solver.pyomo_solver import PyomoSolver

from pyodsp.dec.dd.node_root import DdRootNode
from pyodsp.dec.dd.alg_root_bm import DdAlgRootBm
from pyodsp.dec.dd.alg_root_pbm import DdAlgRootPbm
from pyodsp.dec.dd.node_leaf import DdLeafNode
from pyodsp.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodsp.dec.dd.run import DdRun

from utils import get_args, assert_approximately_equal


def create_master(solver="appsi_highs", pbm=False) -> DdRootNode:
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)
    block.x3 = pyo.Var(within=pyo.Reals)
    vars_dn = {1: [block.x1], 2: [block.x2], 3: [block.x3]}

    block.c1 = pyo.Constraint(expr=3 * block.x1 + 2 * block.x2 + 4 * block.x3 == 17)

    if pbm:
        root_alg = DdAlgRootPbm(block, True, "ipopt", vars_dn)
    else:
        root_alg = DdAlgRootBm(block, True, solver, vars_dn)
    root_node = DdRootNode(0, root_alg, solver)
    return root_node


cost = {1: -4, 2: -1, 3: -6}


def create_sub(i, solver="appsi_highs") -> DdLeafNode:
    block = pyo.ConcreteModel()
    block.x = pyo.Var(bounds=(1, 2))
    vars_up = [block.x]

    block.obj = pyo.Objective(expr=cost[i] * block.x, sense=pyo.minimize)

    sub_solver = PyomoSolver(block, solver, vars_up)
    sub_alg = DdAlgLeafPyomo(sub_solver)
    leaf_node = DdLeafNode(i, sub_alg, 0)
    return leaf_node


def main():
    args = get_args()

    master = create_master(args.solver)
    sub_1 = create_sub(1, args.solver)
    sub_2 = create_sub(2, args.solver)
    sub_3 = create_sub(3, args.solver)

    master.add_child(1)
    master.add_child(2)
    master.add_child(3)

    dd_run = DdRun([master, sub_1, sub_2, sub_3], Path("output/dd/equality"))
    dd_run.run()

    assert_approximately_equal(master.alg_root.bm.obj_bound[-1], -21.5)


if __name__ == "__main__":
    main()
