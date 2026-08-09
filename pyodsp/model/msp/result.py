"""Reading a finished multistage run back in the user's own units."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pyodsp.dec.sddp.policy import SddpPolicy


@dataclass
class MspResult:
    """The outcome of MultistageProgram.solve.

    Attributes:
        name: The program's name.
        is_maximize: Whether the objective was maximized.
        bound: The root's final bound on the optimum, in your units. SDDP
            approaches the optimum from one side and estimates the other
            by simulation, so this is the bound — not a proven optimum.
        first_stage: The here-and-now decision, nested by variable name.
        first_stage_flat: The same, flat, one entry per state position.
        history: The root master's per-iteration trajectory.
        simulation: One row per convergence test — the bound against a
            confidence interval for the simulated policy. SDDP estimates
            that side rather than computing it, so this is what stands in
            for an incumbent. None if the run recorded none.
        num_stages: How many stages the lattice had.
        nodes_per_stage: How many realizations each stage had.
        simulation_samples: The individual objective values behind the
            last convergence test — the empirical distribution its
            interval summarizes. None if the run recorded none.
        lattice: The scenario structure the run was built on.
        is_root_rank: Whether this result carries the answer. Always true
            without MPI. Under MPI every rank returns a result but only
            rank 0 drives the iteration, so the others come back with an
            empty bound and history — check this before reporting.
        output_dir: Where the run's per-node files were written.
    """

    name: str
    is_maximize: bool
    bound: float | None
    first_stage: Dict[str, Any]
    first_stage_flat: Dict[str, float]
    labels: List[str]
    history: pd.DataFrame
    simulation: pd.DataFrame | None
    num_stages: int
    nodes_per_stage: List[int]
    is_root_rank: bool
    output_dir: Path
    simulation_samples: Any = None
    lattice: Any = None
    _nodes: List[List[Any]] = field(default_factory=list, repr=False)

    def policy(self) -> SddpPolicy:
        """The decision rule the run left behind.

        SDDP's answer is not a schedule but a policy: each stage's cuts
        describe what the future costs, so any scenario path can be walked
        after the fact by solving stage by stage against them.

            path = result.policy().simulate(["0-0", "1-2", "2-0"])
        """
        return SddpPolicy(self._nodes)

    def simulate(self, path: List[str]):
        """Walk one scenario path through the policy."""
        return self.policy().simulate(path)

    def summary(self) -> str:
        if not self.is_root_rank:
            return (
                f"{self.name}: this is not rank 0, which is the rank that "
                "drives the iteration and holds the answer."
            )
        sense = "maximize" if self.is_maximize else "minimize"
        lines = [
            f"{self.name}: {sense} via sddp",
            f"  bound      : {_fmt(self.bound)}",
            f"  stages     : {self.num_stages} (nodes per stage "
            f"{self.nodes_per_stage})",
            f"  iterations : {len(self.history)}",
            "  first stage:",
        ]
        for label, value in self.first_stage_flat.items():
            lines.append(f"    {label} = {_fmt(value)}")
        return "\n".join(lines)

    def first_stage_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "variable": list(self.first_stage_flat),
                "value": list(self.first_stage_flat.values()),
            }
        )

    def plot_convergence(self, path=None, *, theme: str = "light"):
        """The root master's trajectory. Requires matplotlib."""
        from pyodsp.viz.convergence import plot_convergence

        if path is None:
            path = Path(self.output_dir) / "convergence.png"
        return plot_convergence(
            self.history,
            path,
            simulation=self.simulation,
            title=f"{self.name} — convergence",
            subtitle=f"SDDP · {self.num_stages} stages · "
            + ("maximize" if self.is_maximize else "minimize"),
            theme=theme,
        )

    def objective_stats(self) -> pd.Series:
        """What the simulated paths cost, summarized.

        The policy is chosen on its expectation; this is the sample that
        expectation was taken over, so it also says how wide and how
        skewed the outcomes behind it are.
        """
        if self.simulation_samples is None or len(self.simulation_samples) == 0:
            raise ValueError(
                "This run recorded no simulation samples. Only SDDP produces "
                "them, and only once it has tested convergence at least once."
            )
        samples = pd.Series(list(self.simulation_samples), dtype=float)
        stats = samples.describe()
        stats["bound"] = self.bound
        if self.simulation is not None and not self.simulation.empty:
            row = self.simulation.iloc[-1]
            stats["ci_lower"] = row["lower"]
            stats["ci_upper"] = row["upper"]
            stats["confidence_level"] = row["confidence_level"]
        return stats

    def plot_objective_distribution(self, path=None, *, bins=None, theme="light"):
        """The empirical distribution of the simulated objective.

        Written into the run's output directory unless told otherwise, as
        the other plot_* methods here are.
        """
        from .viz import plot_objective_distribution

        if path is None:
            path = Path(self.output_dir) / "objective_distribution.png"
        return plot_objective_distribution(self, path, bins=bins, theme=theme)

    def plot_scenario_lattice(self, path=None, *, theme: str = "light"):
        """The scenario structure the run was built on."""
        from .viz import plot_scenario_lattice

        if path is None:
            path = Path(self.output_dir) / "scenario_lattice.png"
        return plot_scenario_lattice(self.lattice, path, name=self.name, theme=theme)

    def plot_state_trajectory(self, paths, variable, path=None, *, theme="light"):
        """One state variable over the horizon, per scenario path."""
        from .viz import plot_state_trajectory

        return plot_state_trajectory(self, paths, variable, path, theme=theme)

    def plot(self, paths, directory=None, *, theme: str = "light"):
        """Every chart that applies. Requires matplotlib.

        Args:
            paths: Named scenario paths to draw trajectories for, each a
                list of node ids as `simulate` takes. A policy has no
                single trajectory of its own, so which paths to show is a
                choice only the caller can make.
            directory: Where to write them; the output directory by
                default.
        """
        from .viz import plot_all

        return plot_all(self, paths, directory or self.output_dir, theme=theme)

    def __repr__(self) -> str:
        return (
            f"MspResult(name={self.name!r}, stages={self.num_stages}, "
            f"bound={self.bound!r})"
        )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def read_result(program, built) -> MspResult:
    """Assemble an MspResult from the solved lattice."""
    root = built.root
    master = built.masters[root.get_idx()]
    values = [var.value for var in master.get_vars()]
    flat = {
        label: value for label, value in zip(built.labels, values) if value is not None
    }

    lattice = program.lattice
    # Under MPI only rank 0 runs the iteration and writes output, so only
    # its masters hold a trajectory to read.
    is_root_rank = program.is_root_rank
    return MspResult(
        name=program.name,
        is_maximize=program.is_maximize,
        bound=root.alg_root.bm.get_objective_bound(),
        first_stage=_nest(built.labels, values),
        first_stage_flat=flat,
        labels=list(built.labels),
        history=(
            _read_history(program.output_dir, root.get_idx())
            if is_root_rank
            else pd.DataFrame(columns=["iteration", "bound", "incumbent"])
        ),
        simulation=_read_simulation(program.output_dir) if is_root_rank else None,
        simulation_samples=(
            _read_simulation_samples(program.output_dir) if is_root_rank else None
        ),
        num_stages=lattice.num_stages,
        nodes_per_stage=[lattice.stage_size(s) for s in range(lattice.num_stages)],
        is_root_rank=is_root_rank,
        output_dir=Path(program.output_dir),
        lattice=lattice,
        _nodes=built.nodes,
    )


def _nest(labels: List[str], values: List[float | None]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for label, value in zip(labels, values):
        if "[" not in label:
            nested[label] = value
            continue
        name, _, index = label.partition("[")
        nested.setdefault(name, {})[index.rstrip("]")] = value
    return nested


def _read_simulation(output_dir):
    from pyodsp.viz.convergence import read_simulation

    return read_simulation(output_dir)


def _read_simulation_samples(output_dir):
    from pyodsp.viz.convergence import read_simulation_samples

    return read_simulation_samples(output_dir)


def _read_history(output_dir, root_idx) -> pd.DataFrame:
    from pyodsp.viz.convergence import read_trajectory

    try:
        return read_trajectory(Path(output_dir) / f"node{root_idx}")
    except FileNotFoundError:
        return pd.DataFrame(columns=["iteration", "bound", "incumbent"])
