from pathlib import Path
import numpy as np
import pyomo.environ as pyo

from balance import first_stage, mid_stage, last_stage, ConstantParams, ScenarioParams
from scenarios import BalanceParams, create_static_scenarios

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf, DecNodeInner
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.sddp.run import SddpRun
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig


def solve(params: BalanceParams, solver="appsi_highs"):
    nodes = []
    for stage in range(params.cp.num_stages):
        node_list = []
        if stage == 0:
            node_id = "0-0"
            node = create_root(
                node_id,
                params.cp,
                params.tp_dict[0][0, :],
                params.init_level,
                solver,
            )
            node_list.append(node)
        elif stage == params.cp.num_stages - 1:
            for s, sp in enumerate(params.sp_dict[stage]):
                node_id = f"{stage}-{s}"
                node = create_leaf(
                    node_id,
                    params.cp,
                    sp,
                    params.final_level,
                    params.final_penalty,
                    solver,
                )
                node_list.append(node)
        else:
            for s, sp in enumerate(params.sp_dict[stage]):
                node_id = f"{stage}-{s}"
                node = create_inner(
                    node_id, stage, params.cp, sp, params.tp_dict[stage][s, :], solver
                )
                node_list.append(node)
        nodes.append(node_list)

    sddp_run = SddpRun(nodes, Path("output/balance/sddp"))
    sddp_run.run()


def create_root(
    idx,
    cp: ConstantParams,
    tp: np.ndarray,
    init_level: float,
    solver_name,
):
    model = pyo.ConcreteModel()
    first_stage(model, init_level, cp)
    coupling_dn = [model.level[cp.time - 1]]
    for t in range(cp.time):
        coupling_dn.append(model.power[t])
    config = SolverConfig(solver_name=solver_name)
    solver_root = PyomoSolver(model, config, coupling_dn)
    alg_root = BdAlgRootBm(solver_root)
    node = DecNodeRoot(idx, alg_root)
    group = []
    for s in range(cp.num_scenarios):
        child_id = f"1-{s}"
        node.add_child(child_id, multiplier=tp[s])
        group.append(child_id)
    node.set_groups([group])
    return node


def create_inner(
    idx,
    stage: int,
    cp: ConstantParams,
    sp: ScenarioParams,
    tp: np.ndarray,
    solver_name,
):
    model = pyo.ConcreteModel()
    model.current_level = pyo.Var()
    model.planned_power = pyo.Var(range(cp.time))
    mid_stage(model, model.current_level, model.power, sp, cp)
    model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.minimize)
    coupling_up = [model.current_level]
    for t in range(cp.time):
        coupling_dn.append(model.planned_power[t])
    coupling_dn = [model.level[cp.time - 1]]
    for t in range(cp.time):
        coupling_dn.append(model.power[t])
    config = SolverConfig(solver_name=solver_name)
    solver_root = PyomoSolver(model, config, coupling_dn)
    alg_root = BdAlgRootBm(solver_root, max_iteration=1)
    solver_leaf = PyomoSolver(model, config, coupling_up)
    alg_leaf = BdAlgLeafPyomo(solver_leaf)
    node = DecNodeInner(idx, alg_root, alg_leaf, log_level=0)
    node.set_bound(-1e6)
    group = []
    for s in range(cp.num_scenarios):
        child_id = f"{stage + 1}-{s}"
        node.add_child(child_id, multiplier=tp[s])
        group.append(child_id)
    node.set_groups([group])
    return node


def create_leaf(
    idx,
    cp: ConstantParams,
    sp: ScenarioParams,
    final_level: float,
    final_penalty: float,
    solver_name,
):
    model = pyo.ConcreteModel()
    model.current_level = pyo.Var()
    model.planned_power = pyo.Var(range(cp.time))
    last_stage(
        model,
        model.current_level,
        final_level,
        final_penalty,
        model.planned_power,
        sp,
        cp,
    )
    model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.minimize)
    coupling_up = [model.current_level]
    for t in range(cp.time):
        coupling_dn.append(model.planned_power[t])
    config = SolverConfig(solver_name=solver_name)
    solver_leaf = PyomoSolver(model, config, coupling_up)
    alg_leaf = BdAlgLeafPyomo(solver_leaf)
    node = DecNodeLeaf(idx, alg_leaf)
    node.set_bound(-1e6)
    return node


if __name__ == "__main__":
    from regime import (
        NormalRegime,
        HotSunnyRegime,
        HotCloudyRegime,
        ColdSunnyRegime,
        ColdCloudyRegime,
        RegimeParams,
    )

    time = 48
    r1 = NormalRegime(time)
    r2 = HotSunnyRegime(time)
    regimes = [r1, r2]
    tm = np.asarray([[0.8, 0.2], [0.6, 0.4]])
    regime_params = RegimeParams(regimes, tm)
    params = create_static_scenarios(3, [1, 1], regime_params, 0)
    solve(params)
