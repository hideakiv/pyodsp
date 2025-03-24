from pathlib import Path
import pyomo.environ as pyo

from aircon import first_stage, mid_stage, last_stage
from utils import assert_approximately_equal

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.node_inner import BdInnerNode
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.run import BdRun
from pyodsp.solver.pyomo_solver import PyomoSolver

def main(solver="appsi_highs", agg=False):
    demand = [1, 1, 3, 1, 3, 1, 3]
    nodes = []
    for idx in range(7):
        if idx == 0:
            node = create_root(idx, demand[idx], solver, agg)
        elif idx <= 2:
            node = create_inner(idx, demand[idx], solver, agg)
        else:
            node = create_leaf(idx, demand[idx], solver)
        nodes.append(node)

    bd_run = BdRun(nodes, Path("output/aircon/bd"))
    bd_run.run()
    assert_approximately_equal(nodes[0].alg_root.bm.obj_bound[-1], 6.25)

def create_root(idx, demand, solver_name, agg=False):
    model = pyo.ConcreteModel()
    first_stage(model, demand)
    coupling_dn = [model.next_inventory]
    solver_root = PyomoSolver(model, solver_name, coupling_dn)
    alg_root = BdAlgRootBm(solver_root)
    node = DecNodeRoot(idx, alg_root)
    node.add_child(1, multiplier=0.5)
    node.add_child(2, multiplier=0.5)
    if agg:
        node.set_groups([[1, 2]])
    return node

def create_inner(idx, demand, solver_name, agg=False):
    model = pyo.ConcreteModel()
    model.prev_inventory = pyo.Var()
    mid_stage(model, model.prev_inventory, demand)
    model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.minimize)
    coupling_up = [model.prev_inventory]
    coupling_dn = [model.next_inventory]
    solver_root = PyomoSolver(model, solver_name, coupling_dn)
    alg_root = BdAlgRootBm(solver_root, max_iteration=1)
    solver_leaf = PyomoSolver(model, solver_name, coupling_up)
    alg_leaf = BdAlgLeafPyomo(solver_leaf)
    parent = (idx - 1) // 2
    node = BdInnerNode(idx, alg_root, alg_leaf, 0, parent)
    node.add_child(2 * idx + 1, multiplier=0.5)
    node.add_child(2 * idx + 2, multiplier=0.5)
    if agg:
        node.set_groups([[2 * idx + 1, 2 * idx + 2]])
    return node

def create_leaf(idx, demand, solver_name):
    model = pyo.ConcreteModel()
    model.prev_inventory = pyo.Var()
    last_stage(model, model.prev_inventory, demand)
    model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.minimize)
    coupling_up = [model.prev_inventory]
    solver_leaf = PyomoSolver(model, solver_name, coupling_up)
    alg_leaf = BdAlgLeafPyomo(solver_leaf)
    parent = (idx - 1) // 2
    node = DecNodeLeaf(idx, alg_leaf)
    node.set_bound(0)
    node.add_parent(parent)
    return node

if __name__ == "__main__":
    main(agg=True)