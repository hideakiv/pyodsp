"""sddpsc against known answers it has to reproduce.

- Two stages: the Caroe-Schultz (1999) instance of examples/bdsc/cs.py,
  driven through Lattice instead of bdsc's HubAndSpoke, has to reach the
  same bound bdsc does.
- Many stages, continuous: where plain SDDP is exact, sddpsc has to land
  on the same bound -- the scaled cut generalizes the Benders cut, it
  does not change the answer.
- Many stages, mixed-integer: the point of sddpsc. An integer inner stage
  plain SDDP cannot price (no duals), and whose LP relaxation is loose,
  converges to the true optimum.
"""

import logging

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.sddpsc.alg_root_bm import SddpScAlgRootBm
from pyodsp.dec.sddpsc.alg_leaf_pyomo import SddpScAlgLeafPyomo
from pyodsp.dec.sddpsc.run import SddpScRun
from pyodsp.model.msp import MultistageProgram

SOLVER = "appsi_highs"

# -- Caroe-Schultz, as in examples/bdsc/cs.py ---------------------------------

R = 2
CS_OPTIMUM = 0.2031  # 3 * (3/4 - 1/32/(1 + R/2)) - 2, to the example's precision


def cs_first_stage(model):
    lb = 1 / 4 - 1 / 32 / (1 + R / 2)
    model.x = pyo.Var(within=pyo.NonNegativeReals, bounds=(lb, 1))


def cs_nodes():
    root_model = pyo.ConcreteModel()
    cs_first_stage(root_model)
    root_model.obj = pyo.Objective(expr=3 * root_model.x)
    root = DecNodeRoot(
        0,
        SddpScAlgRootBm(
            PyomoSolver(root_model, SolverConfig(SOLVER), [root_model.x])
        ),
        log_level_root=0,
    )

    leaves = []
    delta = 1 / 32 / (1 + R / 2)
    for s in range(1, R + 1):
        h = delta * s if s <= R / 2 else 1 / 4 - delta * (s - R / 2)
        model = pyo.ConcreteModel()
        cs_first_stage(model)
        model.y = pyo.Var(within=pyo.Binary)
        model.c1 = pyo.Constraint(expr=-model.y / 2 >= h - model.x)
        model.obj = pyo.Objective(expr=-2 * model.y)
        alg = SddpScAlgLeafPyomo(
            PyomoSolver(model, SolverConfig(SOLVER), [model.x]),
            SolverConfig("ipopt"),
        )
        leaf = DecNodeLeaf(s, alg, log_level_leaf=0)
        leaf.set_bound(-2.01)
        root.add_child(s, multiplier=1 / R)
        leaves.append(leaf)
    root.set_groups([[leaf.get_idx() for leaf in leaves]])
    return [[root], leaves]


def test_caroe_schultz_through_the_lattice(tmp_path):
    run = SddpScRun(
        cs_nodes(),
        tmp_path,
        level=logging.WARNING,
        max_iteration=200,
        sample_frequency=10,
        sample_size=10,
    )

    run.run()

    assert run.graph.bound == pytest.approx(CS_OPTIMUM, abs=1e-4)


# -- continuous multistage, against plain SDDP -------------------------------


def storage(tmp_path, method):
    """Buy up to one unit a stage at a random price, store up to 3, sell
    everything held at the end at a random price. Deterministic level
    transition, random costs only -- which sddpsc requires."""
    msp = MultistageProgram(
        f"storage_{method}",
        sense="min",
        solver=SOLVER,
        stage_bound=-3 * 2.5,
        output_dir=tmp_path / method,
        max_iteration=200,
        sample_frequency=10,
        sample_size=20,
        log_level=logging.CRITICAL,
        method=method,
    )

    @msp.stage(state=["x"])
    def stage(m, state, node):
        if node.is_last:
            m.y = pyo.Var(bounds=(0, 3))
            m.sell = pyo.Constraint(expr=m.y <= state.x)
            return -node["sell"] * m.y
        m.u = pyo.Var(bounds=(0, 1))
        m.x = pyo.Var(bounds=(0, 3))
        m.step = pyo.Constraint(expr=m.x == state.x + m.u)
        return node.get("buy", 1.2) * m.u

    msp.set_initial_state(x=0.0)
    buy = [
        {"name": "cheap", "probability": 0.5, "buy": 0.5},
        {"name": "dear", "probability": 0.5, "buy": 2.4},
    ]
    sell = [
        {"name": "low", "probability": 0.5, "sell": 1.5},
        {"name": "high", "probability": 0.5, "sell": 2.5},
    ]
    msp.set_stage_realizations([buy, buy, sell])
    return msp


def test_matches_plain_sddp_where_that_is_exact(tmp_path):
    sddp = storage(tmp_path, "sddp").solve()
    sddpsc = storage(tmp_path, "sddpsc").solve()

    # each unit is worth 2 at the end: buy at 1.2 at stage 0, and at a
    # stage-1/2 node only when it is cheap -- 1.2 + 2 * 0.5 * 0.5 - 2 * 2
    assert sddp.bound == pytest.approx(-2.3, abs=1e-4)
    assert sddpsc.bound == pytest.approx(sddp.bound, abs=1e-4)
    assert sddpsc.first_stage["x"] == pytest.approx(sddp.first_stage["x"])


# -- mixed-integer multistage, where LP-dual cuts fall short ----------------


def cover(tmp_path, method, integer=True, **kw):
    """Claim x0 in [0, 1.5] at a reward of 1 per unit, cover it at stage 1
    with whole units y1 >= x0 at a random cost c, then sell what was
    bought at a random price r. Fixed transitions, random costs only.

    Each unit nets c - r > 0, so y1 = ceil(x0), and the expected cost is
    f(x0) = -x0 + (E c - E r) * ceil(x0) = -x0 + 0.7 * ceil(x0):
    0 at x0 = 0, -0.3 at x0 = 1, -0.1 at x0 = 1.5. Relaxing y1 drops the
    ceiling, f becomes -0.3 * x0, and the relaxation's optimum is -0.45 at
    x0 = 1.5 -- the bound LP-dual cuts can reach. f's convex envelope is
    exact at x0 = 1, which is what scaled cuts recover.
    """
    msp = MultistageProgram(
        f"cover_{method}",
        sense="min",
        solver=SOLVER,
        stage_bound=-1.0,
        output_dir=tmp_path / f"{method}_{integer}",
        max_iteration=200,
        sample_frequency=10,
        sample_size=20,
        log_level=logging.CRITICAL,
        method=method,
        **kw,
    )
    domain = pyo.NonNegativeIntegers if integer else pyo.NonNegativeReals

    @msp.stage(state=["x"])
    def stage(m, state, node):
        if node.is_first:
            m.x = pyo.Var(bounds=(0, 1.5))
            return -1.0 * m.x
        if node.is_last:
            m.z = pyo.Var(bounds=(0, 2))
            m.sell = pyo.Constraint(expr=m.z <= state.x)
            return -node["r"] * m.z
        m.x = pyo.Var(within=domain, bounds=(0, 2))
        m.cover = pyo.Constraint(expr=m.x >= state.x)
        return node["c"] * m.x

    msp.set_initial_state(x=0.0)
    msp.set_stage_realizations(
        [
            [
                {"name": "lo", "probability": 0.5, "c": 0.5},
                {"name": "hi", "probability": 0.5, "c": 1.5},
            ],
            [
                {"name": "lo", "probability": 0.5, "r": 0.2},
                {"name": "hi", "probability": 0.5, "r": 0.4},
            ],
        ]
    )
    return msp


def test_closes_the_gap_lp_dual_cuts_leave_on_an_integer_stage(tmp_path):
    # Plain SDDP cannot price the integer stage at all (a MIP has no
    # duals), so the most it can certify is the relaxation's bound.
    relaxed = cover(tmp_path, "sddp", integer=False).solve()
    assert relaxed.bound == pytest.approx(-0.45, abs=1e-4)

    result = cover(tmp_path, "sddpsc").solve()

    assert result.bound == pytest.approx(-0.3, abs=1e-4)
    assert result.first_stage["x"] == pytest.approx(1.0, abs=1e-6)
    # and the policy achieves it: x0 = 1, y1 = 1, so a path pays
    # -1 + c - r for its own c and r
    paths = {round(-1 + c - r, 6) for c in (0.5, 1.5) for r in (0.2, 0.4)}
    samples = {round(v, 6) for v in result.simulation_samples}
    assert samples <= paths
