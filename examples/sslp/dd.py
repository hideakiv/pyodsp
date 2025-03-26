from pathlib import Path
import pyomo.environ as pyo

from sslp import first_stage, second_stage

from pyodsp.dec.node.dec_node import DecNodeLeaf
from pyodsp.dec.dd.node_root import DdRootNode
from pyodsp.dec.dd.alg_root_bm import DdAlgRootBm
from pyodsp.dec.dd.alg_root_pbm import DdAlgRootPbm
from pyodsp.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodsp.dec.dd.run import DdRun
from pyodsp.solver.pyomo_solver import PyomoSolver

def main(nI: int, nJ: int, nS: int, solver="appsi_highs"):
    nodes = []
    master = create_master(nJ, nS, solver)

    nodes.append(master)

    for s in range(nS):
        sub = create_sub(s, nI, nJ, nS, solver)
        master.add_child(s+1)
        nodes.append(sub)

    dd_run = DdRun(nodes, Path("output/sslp/dd"))
    dd_run.run()

def create_master(nJ: int, nS: int, solver="appsi_highs", pbm=False) -> DdRootNode:
    m = pyo.ConcreteModel()
    m.sJ = pyo.RangeSet(nJ)
    m.sS = pyo.RangeSet(nS)
    m.x = pyo.Var(m.sJ, m.sS, domain=pyo.Binary)
    vars_dn = {}
    for s in m.sS:
        xs = [m.x[j,s] for j in m.sJ]
        vars_dn[s] = xs

    def rule_x(m, j, s):
        if s == nS:
            return m.x[j, s] == m.x[j, 1]
        else:
            return m.x[j, s] == m.x[j, s+1]
    m.constr = pyo.Constraint(m.sJ, m.sS, rule=rule_x)
    
    if pbm:
        root_alg = DdAlgRootPbm(m, True, "ipopt", vars_dn)
    else:
        root_alg = DdAlgRootBm(m, True, solver, vars_dn)
    root_node = DdRootNode(0, root_alg, solver)
    return root_node

def create_sub(s: int, nI: int, nJ: int, nS: int, solver="appsi_highs") -> DecNodeLeaf:
    m = pyo.ConcreteModel()
    first_stage(m, nJ)
    m.b = pyo.Block()
    second_stage(m.b, m.x, nI, nJ, seed2=3 + s)
    m.obj.expr = (m.obj.expr + m.b.objexpr) / nS

    vars_up = [m.x[j] for j in range(1, nJ+1)]

    sub_solver = PyomoSolver(m, solver, vars_up)
    sub_alg = DdAlgLeafPyomo(sub_solver)
    leaf_node = DecNodeLeaf(s+1, sub_alg)
    leaf_node.add_parent(0)

    return leaf_node

if __name__ == "__main__":
    main(50, 10, 5)