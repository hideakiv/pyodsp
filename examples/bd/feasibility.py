from pathlib import Path

import pyomo.environ as pyo

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.run import BdRun

from utils import get_args, assert_approximately_equal


def create_root_node(solver="appsi_highs"):
    model1 = pyo.ConcreteModel()

    model1.x1 = pyo.Var(within=pyo.NonNegativeReals)
    model1.x2 = pyo.Var(within=pyo.NonNegativeReals)

    model1.obj = pyo.Objective(expr=3 * model1.x1 + 2 * model1.x2, sense=pyo.minimize)

    coupling_dn = [model1.x1, model1.x2]
    config = SolverConfig(solver_name=solver)
    first_stage_solver = PyomoSolver(model1, config, coupling_dn)
    first_stage_alg = BdAlgRootBm(first_stage_solver)
    root_node = DecNodeRoot(0, first_stage_alg)
    return root_node


xi1 = {1: 4, 2: 6}
xi2 = {1: 4, 2: 8}
p = {1: 0.5, 2: 0.5}


def create_leaf_node(i, solver="appsi_highs"):
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.NonNegativeReals)
    block.x2 = pyo.Var(within=pyo.NonNegativeReals)

    block.y1 = pyo.Var(within=pyo.NonNegativeReals)
    block.y2 = pyo.Var(within=pyo.NonNegativeReals)

    block.c1 = pyo.Constraint(expr=3 * block.y1 + 2 * block.y2 <= block.x1)
    block.c2 = pyo.Constraint(expr=2 * block.y1 + 5 * block.y2 <= block.x2)
    block.c3 = pyo.Constraint(expr=block.y1 >= 0.8 * xi1[i])
    block.c4 = pyo.Constraint(expr=block.y1 <= xi1[i])
    block.c5 = pyo.Constraint(expr=block.y2 >= 0.8 * xi2[i])
    block.c6 = pyo.Constraint(expr=block.y2 >= xi2[i])

    block.obj = pyo.Objective(expr=-15 * block.y1 - 12 * block.y2, sense=pyo.minimize)

    coupling_up = [block.x1, block.x2]
    config = SolverConfig(solver_name=solver)
    second_stage_solver = PyomoSolver(block, config, coupling_up)
    second_stage_alg = BdAlgLeafPyomo(second_stage_solver)
    leaf_node = DecNodeLeaf(i, second_stage_alg)
    leaf_node.set_bound(-1000.0)
    leaf_node.add_parent(0)
    return leaf_node


def main():
    args = get_args()

    root_node = create_root_node(args.solver)
    leaf_node_1 = create_leaf_node(1, args.solver)
    leaf_node_2 = create_leaf_node(2, args.solver)

    root_node.add_child(1, multiplier=p[1])
    root_node.add_child(2, multiplier=p[2])

    root_node.set_groups([[1, 2]])

    bd_run = BdRun([root_node, leaf_node_1, leaf_node_2], Path("output/bd/feasibility"))
    bd_run.run()

    assert_approximately_equal(root_node.alg_root.bm.obj_bound[-1], 26.48)


if __name__ == "__main__":
    main()
