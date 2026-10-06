"""The row-generation cut master (cut_master='bm', the default).

van der Laan & Romeijnders (Algorithm 2) compute C_s(rho) by row
generation: the cut-generation master (CGMP) is an LP over (alpha, beta,
tau) whose rows are points of

    S^phi_s = {(x, theta, y) : x in X, theta >= phi(x), y feasible for x},

phi being the master's current outer approximation, and the cut-generation
subproblem (CGSP) finds the most violated point. The returned cut is the
best evaluated (beta, tau) with alpha set to the CGSP's value there, which
makes it valid however early the loop stops.

The instance is Caroe & Schultz with r = 2, as in examples/bdsc/cs.py.
"""

import logging

import numpy as np
import pyomo.environ as pyo
import pytest

from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.alg_root_bm import BdScAlgRootBm
from pyodsp.dec.bdsc.run import BdScRun
from pyodsp.dec.node.dec_node import DecNodeLeaf, DecNodeRoot
from pyodsp.model.sp import StochasticProgram
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

R = 2
DELTA = 1 / 32 / (1 + R / 2)
LOWER = 1 / 4 - DELTA
H = {s: DELTA * s if s <= R / 2 else 1 / 4 - DELTA * (s - R / 2) for s in (1, 2)}
BOUND = -2.01
OPTIMUM = 0.203125


def make_nodes(cut_master="bm", cg_iterations=1000):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(LOWER, 1))
    model.obj = pyo.Objective(expr=3 * model.x, sense=pyo.minimize)
    root = DecNodeRoot(
        0,
        BdScAlgRootBm(PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])),
        log_level_root=logging.CRITICAL,
    )
    nodes = [root]
    for s in (1, 2):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(LOWER, 1))
        m.y = pyo.Var(within=pyo.Binary)
        m.c1 = pyo.Constraint(expr=-m.y / 2 >= H[s] - m.x)
        m.obj = pyo.Objective(expr=-2 * m.y, sense=pyo.minimize)
        leaf = DecNodeLeaf(
            s,
            BdScAlgLeafPyomo(
                PyomoSolver(m, SolverConfig("appsi_highs"), [m.x]),
                SolverConfig("ipopt"),
                cg_iterations,
                cut_master=cut_master,
            ),
            log_level_leaf=logging.CRITICAL,
        )
        leaf.set_bound(BOUND)
        root.add_child(s, multiplier=1 / R)
        nodes.append(leaf)
    root.set_groups([[1, 2]])
    return nodes


def phi_of(alg):
    """The outer approximation the leaf priced against: theta >= its cuts."""
    cuts = [
        (c.cut.rhs, c.cut.coeffs.get(0, 0.0)) for g in alg.cgsp.get_cuts() for c in g
    ]
    floor = alg._subobj_bound

    def phi(x):
        return max([floor] + [rhs - slope * x for rhs, slope in cuts])

    return phi


def points_of_S(h, phi):
    """Points of S^phi_s on a fine grid, with both feasible y."""
    xs = np.unique(np.concatenate([np.linspace(LOWER, 1, 2001), [h + 0.5]]))
    for x in xs:
        theta = phi(x)
        yield x, theta, 0.0  # y = 0 always feasible, q = 0
        if x >= h + 0.5 - 1e-12:
            yield x, theta, -2.0  # y = 1, q = -2


def capture_up_messages(monkeypatch):
    seen = []
    original = BdScAlgLeafPyomo.get_up_message

    def recording(self):
        message = original(self)
        seen.append((self.idx, phi_of(self), self._x_hat, self._rho, message))
        return message

    monkeypatch.setattr(BdScAlgLeafPyomo, "get_up_message", recording)
    return seen


@pytest.mark.parametrize("cg_iterations", [2, 1000], ids=["cut-short", "converged"])
def test_every_cut_is_valid_on_the_whole_of_S_phi(tmp_path, monkeypatch, cg_iterations):
    """q(y) >= alpha - beta x - tau theta at every point the cut must hold
    for — the guarantee the CGSP correction of alpha provides.

    Cut short, the CGMP's own alpha is too optimistic: it has only seen the
    columns generated so far. Converged, the two agree, so the short loop is
    the case that tests the correction."""
    seen = capture_up_messages(monkeypatch)
    nodes = make_nodes(cg_iterations=cg_iterations)
    BdScRun(nodes, tmp_path, level=logging.CRITICAL, max_iteration=50).run()

    assert seen
    for idx, phi, _, _, message in seen:
        cut = message.get_cut()
        alpha, beta, tau = cut.rhs, cut.coeffs.get(0, 0.0), message.get_tau()
        assert tau >= -1e-9
        worst = min(
            q - (alpha - beta * x - tau * theta)
            for x, theta, q in points_of_S(H[idx], phi)
        )
        assert worst >= -1e-6, f"scenario {idx}: cut violated by {-worst:.3g}"


def test_c_is_the_value_of_the_returned_cut(tmp_path, monkeypatch):
    """c = alpha - beta x_hat - rho (1 + tau): what the root's rho update
    and its 'step the master' test read is the cut it is sent, not a model
    estimate."""
    seen = capture_up_messages(monkeypatch)
    BdScRun(make_nodes(), tmp_path, level=logging.CRITICAL, max_iteration=50).run()

    for _, _, x_hat, rho, message in seen:
        cut = message.get_cut()
        alpha, beta, tau = cut.rhs, cut.coeffs.get(0, 0.0), message.get_tau()
        assert message.get_c() == pytest.approx(
            alpha - beta * x_hat[0] - rho * (1 + tau), abs=1e-9
        )


@pytest.mark.parametrize("cut_master", ["bm", "pbm"])
def test_both_cut_masters_solve_the_instance(tmp_path, cut_master):
    sp = StochasticProgram(
        "cs",
        sense="min",
        method="bdsc",
        cut_master=cut_master,
        solver="appsi_highs",
        recourse_bound=BOUND,
        output_dir=tmp_path,
        log_level=logging.CRITICAL,
    )

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(LOWER, 1))
        return 3 * m.x

    @sp.recourse
    def recourse(m, state, scenario):
        m.y = pyo.Var(within=pyo.Binary)
        m.c1 = pyo.Constraint(expr=-m.y / 2 >= scenario["h"] - state.x)
        return -2 * m.y

    sp.set_scenarios({f"s{s}": {"h": h} for s, h in H.items()})
    result = sp.solve()

    assert result.objective == pytest.approx(OPTIMUM, abs=1e-6)


def test_an_unknown_cut_master_is_rejected():
    with pytest.raises(ValueError, match="cut_master"):
        StochasticProgram("cs", cut_master="lp")


def test_the_row_generation_master_is_an_lp_on_the_subproblems_solver():
    """No quadratic term, so no QP solver: it borrows the subproblem's."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.obj = pyo.Objective(expr=m.x)
    alg = BdScAlgLeafPyomo(
        PyomoSolver(m, SolverConfig("appsi_highs"), [m.x]), SolverConfig("ipopt")
    )

    assert alg.mc.solver_config.solver_name == "appsi_highs"
