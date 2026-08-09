"""The scenario structure of a multistage program.

SDDP here works on a *recombining* lattice, not a scenario tree: each
stage has its own set of realizations, and every node at stage t leads to
every node at stage t+1 with a transition probability. That is what keeps
the problem size linear in the horizon — a tree would multiply out.

Two shapes cover most models:

- stage-wise independent: the same realizations every stage, drawn with
  the same probabilities regardless of what came before. `independent`
  builds this, and it is what most textbook multistage models assume.
- Markov: the realizations are states of a chain, so where you go next
  depends on where you are. `ScenarioLattice` takes those transitions
  directly.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..scenario import Scenario, as_scenario_set

PROBABILITY_TOL = 1e-9


@dataclass(frozen=True)
class LatticeNode:
    """One realization at one stage, as the stage builder sees it.

    Attributes:
        stage: Which stage it belongs to, counting from 0.
        index: Its position within that stage.
        name: Its name, unique within the stage.
        data: The realized values, handed to the builder untouched.
        is_first: True at stage 0, whose state is the initial condition
            rather than a decision of an earlier stage.
        is_last: True at the final stage, which passes no state on.
    """

    stage: int
    index: int
    name: str
    data: Dict[str, Any]
    is_first: bool
    is_last: bool

    @property
    def idx(self) -> str:
        """The node id the decomposition layer uses."""
        return f"{self.stage}-{self.index}"

    def __getitem__(self, key: str) -> Any:
        try:
            return self.data[key]
        except KeyError:
            if self.is_first and not self.data:
                raise KeyError(
                    f"Node {self.idx} has no data. The first stage is decided "
                    "before anything is observed, so it has no realization of "
                    f"its own - reach for {key!r} with node.get({key!r}, "
                    "default), branch on node.is_first, or give the stage "
                    "known values with first_stage_data={...}."
                ) from None
            raise KeyError(
                f"Node {self.idx} has no data entry {key!r}; it has "
                f"{sorted(self.data)}"
            ) from None

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getattr__(self, key: str) -> Any:
        try:
            return self.data[key]
        except KeyError:
            raise AttributeError(
                f"Node {self.idx} has no data entry {key!r}; it has "
                f"{sorted(self.data)}"
            ) from None


class ScenarioLattice:
    """Realizations per stage, plus how one stage's lead to the next's.

    Args:
        stages: The realizations at each stage, outermost list indexed by
            stage. Stage 0 must hold exactly one — the present is not
            uncertain.
        transitions: `transitions[t][i][j]` is the probability of moving
            from node i of stage t to node j of stage t+1. One matrix per
            stage except the last, each row summing to one.
    """

    def __init__(
        self,
        stages: Sequence[Sequence[Scenario]],
        transitions: Sequence[Sequence[Sequence[float]]],
    ) -> None:
        self._stages = [list(stage) for stage in stages]
        self._transitions = [[list(row) for row in matrix] for matrix in transitions]
        self._validate()

    def _validate(self) -> None:
        if len(self._stages) < 2:
            raise ValueError(
                f"A multistage program needs at least two stages, got "
                f"{len(self._stages)}. Use pyodsp.model.sp.StochasticProgram "
                "for a two-stage problem stated as here-and-now plus recourse."
            )
        if len(self._stages[0]) != 1:
            raise ValueError(
                f"Stage 0 must hold exactly one node, got "
                f"{len(self._stages[0])}. The first stage is decided before "
                "anything is observed, so it has no realizations of its own."
            )
        for stage, nodes in enumerate(self._stages):
            if not nodes:
                raise ValueError(f"Stage {stage} has no nodes")
            names = [node.name for node in nodes]
            duplicates = {name for name in names if names.count(name) > 1}
            if duplicates:
                raise ValueError(
                    f"Stage {stage} has duplicate node names {sorted(duplicates)}"
                )

        expected = len(self._stages) - 1
        if len(self._transitions) != expected:
            raise ValueError(
                f"Expected {expected} transition matrices for "
                f"{len(self._stages)} stages, got {len(self._transitions)}. "
                "The last stage transitions to nothing."
            )
        for stage, matrix in enumerate(self._transitions):
            rows, columns = len(self._stages[stage]), len(self._stages[stage + 1])
            if len(matrix) != rows:
                raise ValueError(
                    f"Transition matrix {stage} has {len(matrix)} rows but "
                    f"stage {stage} has {rows} nodes"
                )
            for i, row in enumerate(matrix):
                if len(row) != columns:
                    raise ValueError(
                        f"Row {i} of transition matrix {stage} has {len(row)} "
                        f"entries but stage {stage + 1} has {columns} nodes"
                    )
                if any(probability < 0.0 for probability in row):
                    raise ValueError(
                        f"Row {i} of transition matrix {stage} has a negative "
                        "probability"
                    )
                total = sum(row)
                if abs(total - 1.0) > PROBABILITY_TOL:
                    raise ValueError(
                        f"Row {i} of transition matrix {stage} sums to {total}, "
                        "not 1. Each node's transitions are a distribution over "
                        "the next stage."
                    )

    # -- access ------------------------------------------------------------

    @property
    def num_stages(self) -> int:
        return len(self._stages)

    def stage_size(self, stage: int) -> int:
        return len(self._stages[stage])

    def nodes(self, stage: int) -> List[LatticeNode]:
        """The stage's realizations, as the builder sees them."""
        last = self.num_stages - 1
        return [
            LatticeNode(
                stage=stage,
                index=index,
                name=scenario.name,
                data=scenario.data,
                is_first=stage == 0,
                is_last=stage == last,
            )
            for index, scenario in enumerate(self._stages[stage])
        ]

    def all_nodes(self) -> List[LatticeNode]:
        return [node for stage in range(self.num_stages) for node in self.nodes(stage)]

    def transitions_from(self, stage: int, index: int) -> List[float]:
        """Where node `index` of `stage` can go, and how likely each is."""
        return list(self._transitions[stage][index])

    def reaching_probability(self) -> List[List[float]]:
        """The chance of standing at each node, stage by stage.

        Not needed to solve — SDDP only ever uses one stage's transitions
        at a time — but it is what makes a lattice readable, and what a
        summary reports.
        """
        distribution = [[1.0]]
        for stage in range(self.num_stages - 1):
            current = distribution[-1]
            nxt = [0.0] * self.stage_size(stage + 1)
            for i, mass in enumerate(current):
                for j, probability in enumerate(self._transitions[stage][i]):
                    nxt[j] += mass * probability
            distribution.append(nxt)
        return distribution

    def __repr__(self) -> str:
        sizes = [self.stage_size(s) for s in range(self.num_stages)]
        return f"ScenarioLattice(stages={self.num_stages}, sizes={sizes})"


def independent(
    realizations,
    num_stages: int,
    *,
    first_stage_name: str = "root",
    first_stage_data: Mapping[str, Any] | None = None,
    normalize: bool = False,
) -> ScenarioLattice:
    """A lattice whose realizations do not depend on the past.

    The same set is drawn at every stage after the first, with the same
    probabilities — so every row of every transition matrix is that one
    distribution.

    Args:
        realizations: The per-stage realizations, in any form
            `as_scenario_set` accepts.
        num_stages: The horizon, counting the first stage. Four means one
            here-and-now decision followed by three observed stages.
        first_stage_name: Name for the single stage-0 node.
        normalize: Rescale the realization probabilities to sum to one.

    Returns:
        A ScenarioLattice.
    """
    if num_stages < 2:
        raise ValueError(
            f"num_stages must be at least 2, got {num_stages}. With one stage "
            "there is nothing uncertain to model."
        )
    scenarios = as_scenario_set(realizations, normalize=normalize)
    probabilities = list(scenarios.probabilities)

    stages: List[List[Scenario]] = [
        [Scenario(first_stage_name, 1.0, dict(first_stage_data or {}))]
    ]
    stages.extend([list(scenarios) for _ in range(num_stages - 1)])

    transitions = [
        [list(probabilities) for _ in range(len(stages[stage]))]
        for stage in range(num_stages - 1)
    ]
    return ScenarioLattice(stages, transitions)


def stage_varying(
    per_stage,
    *,
    transitions: Sequence[Sequence[Sequence[float]]] | None = None,
    first_stage_name: str = "root",
    first_stage_data: Mapping[str, Any] | None = None,
) -> ScenarioLattice:
    """A lattice whose realizations change from stage to stage.

    The distribution may differ at every stage, and so may the number of
    realizations - a model whose uncertainty grows or narrows over time is
    described here rather than through `independent`.

    Args:
        per_stage: One collection of realizations per stage *after* the
            first, each in any form as_scenario_set accepts.
        transitions: One matrix per transition. Left out, each stage's
            realizations are drawn independently of the previous stage,
            with that stage's own probabilities.
        first_stage_name: Name for the single stage-0 node.
        first_stage_data: Known data for the first stage, which observes
            nothing but may still have values the builder needs.
    """
    per_stage = [as_scenario_set(stage) for stage in per_stage]
    if not per_stage:
        raise ValueError(
            "per_stage is empty; give the realizations of each stage after "
            "the first."
        )

    stages: List[List[Scenario]] = [
        [Scenario(first_stage_name, 1.0, dict(first_stage_data or {}))]
    ]
    stages.extend([list(scenarios) for scenarios in per_stage])

    if transitions is None:
        # Independent between stages, but each stage keeps its own
        # distribution - every row of a matrix is that stage's.
        transitions = [
            [list(per_stage[stage].probabilities) for _ in range(len(stages[stage]))]
            for stage in range(len(stages) - 1)
        ]
    return ScenarioLattice(stages, transitions)


def markov(
    realizations,
    num_stages: int,
    transition_matrix: Sequence[Sequence[float]],
    *,
    initial_distribution: Sequence[float] | None = None,
    first_stage_name: str = "root",
    first_stage_data: Mapping[str, Any] | None = None,
) -> ScenarioLattice:
    """A lattice whose realizations are the states of a Markov chain.

    Args:
        realizations: The chain's states, one per realization.
        num_stages: The horizon, counting the first stage.
        transition_matrix: Square, row-stochastic, one row per state.
        initial_distribution: How stage 1 is reached from the first stage.
            The realization probabilities are used when omitted.
        first_stage_name: Name for the single stage-0 node.
    """
    if num_stages < 2:
        raise ValueError(f"num_stages must be at least 2, got {num_stages}")
    scenarios = as_scenario_set(realizations)
    size = len(scenarios)

    if len(transition_matrix) != size:
        raise ValueError(
            f"transition_matrix has {len(transition_matrix)} rows but there "
            f"are {size} realizations"
        )
    initial = (
        list(initial_distribution)
        if initial_distribution is not None
        else list(scenarios.probabilities)
    )
    if len(initial) != size:
        raise ValueError(
            f"initial_distribution has {len(initial)} entries but there are "
            f"{size} realizations"
        )

    stages: List[List[Scenario]] = [
        [Scenario(first_stage_name, 1.0, dict(first_stage_data or {}))]
    ]
    stages.extend([list(scenarios) for _ in range(num_stages - 1)])

    transitions: List[List[List[float]]] = [[initial]]
    transitions.extend(
        [[list(row) for row in transition_matrix] for _ in range(num_stages - 2)]
    )
    return ScenarioLattice(stages, transitions)


def as_lattice(value, num_stages: int | None = None) -> ScenarioLattice:
    """Coerce the accepted spellings into a ScenarioLattice."""
    if isinstance(value, ScenarioLattice):
        return value
    if num_stages is None:
        raise ValueError(
            "num_stages is needed to build a lattice from realizations; pass "
            "a ScenarioLattice to describe the stages yourself."
        )
    if isinstance(value, (Mapping, Iterable)):
        return independent(value, num_stages)
    raise TypeError(
        "realizations must be a ScenarioLattice or something " "as_scenario_set accepts"
    )
