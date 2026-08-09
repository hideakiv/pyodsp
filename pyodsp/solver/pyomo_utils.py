from pathlib import Path
from typing import List

import pandas as pd
from pyomo.environ import Var, Objective, ScalarVar, minimize

from pyodsp.solver.pyomo_solver import PyomoSolver

# Re-exported: this was defined here before PyomoSolver needed it, and
# callers still import it from this module.
from pyodsp.solver.sense import negate_objective_sense  # noqa: F401


def negate_saved_objective_csv(node_dir: Path) -> None:
    """Negate the obj_bound/obj_val/center_val columns of the bm.csv or
    pbm.csv that BundleMethod/ProximalBundleMethod/RestrictedBundleMethod
    saved for one node (under `<run filedir>/node<idx>/`).

    The algorithms now save these already in the user's units — a
    maximize model is converted by PyomoSolver, which remembers the flip,
    and BundleMethod.save undoes it on the way out. This remains for
    correcting a file produced some other way.
    """
    node_dir = Path(node_dir)
    for filename in ("bm.csv", "pbm.csv"):
        path = node_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        for col in ("obj_bound", "obj_val", "center_val"):
            if col in df.columns:
                df[col] = -df[col]
        df.to_csv(path)
        return
    raise FileNotFoundError(f"No bm.csv or pbm.csv found in {node_dir}")


def add_linear_terms_to_objective(
    solver: PyomoSolver, coeffs: List[float], vars: Var | List[ScalarVar]
) -> None:
    solver.original_objective.deactivate()
    update_linear_terms_in_objective(solver, coeffs, vars)


def update_linear_terms_in_objective(
    solver: PyomoSolver, coeffs: List[float], vars: Var | List[ScalarVar]
) -> None:
    modified_expr = solver.original_objective.expr + sum(
        coeffs[i] * vars[i] for i in range(len(coeffs))
    )

    if solver.model.component("_mod_obj") is not None:
        solver.model.del_component("_mod_obj")

    solver.model._mod_obj = Objective(
        expr=modified_expr, sense=solver.original_objective.sense
    )


def add_terms_to_objective(solver: PyomoSolver, vars: Var) -> None:
    coeffs = [1.0 for _ in range(len(vars))]
    add_linear_terms_to_objective(solver, coeffs, vars)


def update_quad_terms_in_objective(
    solver: PyomoSolver,
    quadvars: List[ScalarVar],
    center: List[float],
    penalty: float = 1.0,
) -> None:
    """Add a quadratic proximal penalty on top of `solver.model._mod_obj`.

    Only used by ProximalBundleMethod/RestrictedBundleMethod, whose
    `_mod_obj` is always built (via CuttingPlaneMethod.build_theta_objective)
    as a minimize-sense objective regardless of the true optimization
    direction — so the penalty is always added, never subtracted.
    """
    modified_expr = solver.model._mod_obj.expr + 0.5 * sum(
        penalty * (quadvar - val) ** 2 for quadvar, val in zip(quadvars, center)
    )

    if solver.model.component("_mod_quad_obj") is not None:
        solver.model.del_component("_mod_quad_obj")

    solver.model._mod_quad_obj = Objective(expr=modified_expr, sense=minimize)
