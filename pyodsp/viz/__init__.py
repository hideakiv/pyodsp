"""Charts for pyodsp run artifacts.

Algorithm-agnostic: everything here works from what a run recorded, so it
applies to BD, BDSC, DD and SDDP alike. Charts specific to a modelling
front-end live with that front-end instead (see pyodsp.model.sp.viz).

Needs matplotlib, which pyodsp does not require — `pip install pyodsp[viz]`.

    from pyodsp.viz import plot_run_convergence

    plot_run_convergence("output/bdsc/cs/node0", "convergence.png")
"""

from .convergence import plot_convergence, plot_run_convergence, read_trajectory

__all__ = [
    "plot_convergence",
    "plot_run_convergence",
    "read_trajectory",
]
