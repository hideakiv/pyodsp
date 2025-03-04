from pathlib import Path
import pyomo.environ as pyo

from flowergirl import first_stage, mid_stage, last_stage

from pyodsp.dec.bd.node_root import BdRootNode
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.node_inner import BdInnerNode
from pyodsp.dec.bd.node_leaf import BdLeafNode
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.run import BdRun
from pyodsp.solver.pyomo_solver import PyomoSolver

def main(solver="appsi_highs"):
    demand = [0, 10, 15, 20, 21, 22, 28, 38, 37, 36, 35, 34, 33, 32, 31]
    nodes = []
    for idx in range(15):
        if idx == 0:
            node = create_root(solver)
        elif idx <= 6:
            node = create_inner(idx, demand[idx], solver)
        else:
            node = create_leaf(idx, demand[idx], solver)
        nodes.append(node)

    bd_run = BdRun(nodes, Path("output/flowergirl/bd"))
    bd_run.run()

def create_root(solver_name):
    model = pyo.ConcreteModel()
    first_stage(model)
    coupling_dn = [model.init_purchase]
    solver_root = PyomoSolver(model, solver_name, coupling_dn)
    alg_root = BdAlgRootBm(solver_root)
    node = BdRootNode(0, alg_root)
    node.add_child(1, multiplier=0.5)
    node.add_child(2, multiplier=0.5)
    return node

def create_inner(idx, demand, solver_name):
    model = pyo.ConcreteModel()
    model.prev_inventory = pyo.Var()
    mid_stage(model, model.prev_inventory, demand)
    model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.maximize)
    coupling_up = [model.prev_inventory]
    coupling_dn = [model.next_inventory]
    solver_root = PyomoSolver(model, solver_name, coupling_dn)
    alg_root = BdAlgRootBm(solver_root, max_iteration=2)
    solver_leaf = PyomoSolver(model, solver_name, coupling_up)
    alg_leaf = BdAlgLeafPyomo(solver_leaf)
    parent = (idx - 1) // 2
    node = BdInnerNode(idx, alg_root, alg_leaf, 1000, parent)
    node.add_child(2 * idx + 1, multiplier=0.5)
    node.add_child(2 * idx + 2, multiplier=0.5)
    return node

def create_leaf(idx, demand, solver_name):
    model = pyo.ConcreteModel()
    model.prev_inventory = pyo.Var()
    last_stage(model, model.prev_inventory, demand)
    model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.maximize)
    coupling_up = [model.prev_inventory]
    solver_leaf = PyomoSolver(model, solver_name, coupling_up)
    alg_leaf = BdAlgLeafPyomo(solver_leaf)
    parent = (idx - 1) // 2
    node = BdLeafNode(idx, alg_leaf, 1000, parent)
    return node

if __name__ == "__main__":
    main()