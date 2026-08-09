import pyomo.environ as pyo
from pyodsp.solver.pyomo_solver import SolverConfig
from pyodsp.dec.bdsc.master_creator import MasterCreator as BdScMasterCreator
from pyodsp.dec.dd.master_creator import MasterCreator as DdMasterCreator


def test_bdsc_master_creator_builds_a_maximize_pricing_problem():
    """Pricing is the dual of the restricted master, so it is a maximize
    problem — and must survive as one. PyomoSolver converts maximize
    models by default; this master opts out, leaving it for
    CuttingPlaneMethod's sign flip to solve as a minimization."""
    creator = BdScMasterCreator(solver_config=SolverConfig("appsi_highs"))

    solver = creator.create(solution=[1.0, 2.0], rho=0.5)

    assert solver.model.objective.sense == pyo.maximize
    assert solver.is_minimize() is False
    assert solver.sense_multiplier == 1.0
    assert len(solver.get_vars()) == 3  # tau + one beta per solution entry


def make_coupling_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals)
    model.y = pyo.Var(domain=pyo.Reals)
    # both-sided
    model.con_both = pyo.Constraint(expr=(0, model.x + model.y, 10))
    # lower bound only
    model.con_lb = pyo.Constraint(expr=model.x >= 1)
    # upper bound only
    model.con_ub = pyo.Constraint(expr=model.y <= 5)
    return model


def test_dd_master_creator_fixes_ld_minus_when_no_lower_bound():
    model = make_coupling_model()
    vars_dn = {0: [model.x, model.y]}
    creator = DdMasterCreator(
        model, solver_config=SolverConfig("appsi_highs"), vars_dn=vars_dn
    )

    solver = creator.create()

    ub_only_index = creator.lagrangian_data.constraints.index(model.con_ub)
    assert solver.model.ld_minus[ub_only_index].fixed is True
    assert solver.model.ld_minus[ub_only_index].value == 0


def test_dd_master_creator_fixes_ld_plus_when_no_upper_bound():
    model = make_coupling_model()
    vars_dn = {0: [model.x, model.y]}
    creator = DdMasterCreator(
        model, solver_config=SolverConfig("appsi_highs"), vars_dn=vars_dn
    )

    solver = creator.create()

    lb_only_index = creator.lagrangian_data.constraints.index(model.con_lb)
    assert solver.model.ld_plus[lb_only_index].fixed is True
    assert solver.model.ld_plus[lb_only_index].value == 0


def test_dd_master_creator_does_not_fix_either_when_both_bounds_present():
    model = make_coupling_model()
    vars_dn = {0: [model.x, model.y]}
    creator = DdMasterCreator(
        model, solver_config=SolverConfig("appsi_highs"), vars_dn=vars_dn
    )

    solver = creator.create()

    both_index = creator.lagrangian_data.constraints.index(model.con_both)
    assert solver.model.ld_plus[both_index].fixed is False
    assert solver.model.ld_minus[both_index].fixed is False


def test_dd_master_creator_builds_a_maximize_lagrangian_dual():
    model = make_coupling_model()
    vars_dn = {0: [model.x, model.y]}
    creator = DdMasterCreator(
        model, solver_config=SolverConfig("appsi_highs"), vars_dn=vars_dn
    )

    solver = creator.create()

    # The dual of a minimize problem is a maximize one. That inversion is
    # the point, so this master opts out of PyomoSolver's conversion and
    # CuttingPlaneMethod's sign flip solves it as a minimization instead.
    assert solver.model.objective.sense == pyo.maximize
    assert solver.is_minimize() is False
    assert solver.sense_multiplier == 1.0
    assert len(solver.get_vars()) == creator.num_constrs
