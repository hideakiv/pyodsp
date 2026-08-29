import logging

# The iteration table: a header row of column names, then aligned numbers.
# bm, pbm and rbm share it; pbm and rbm add the CB (center bound) column and
# pbm the u (penalty) column, so the header is drawn from whichever columns
# the row actually carries. Columns appear in this order; a missing value
# renders as "-".
#     key -> (header label, column width, value formatter)
_COLUMN_SPECS = {
    "iteration": ("iter", 6, lambda v: f"{v:d}"),
    "lb": ("LB", 14, lambda v: f"{v:.4f}"),
    "center_bound": ("CB", 14, lambda v: f"{v:.4f}"),
    "ub": ("UB", 14, lambda v: f"{v:.4f}"),
    "num_cuts": ("cuts", 5, lambda v: f"{v:d}"),
    "penalty": ("u", 9, lambda v: f"{v}"),
    "elapsed": ("elapsed", 9, lambda v: f"{v:.2f}"),
}
_COLUMN_ORDER = ("iteration", "lb", "center_bound", "ub", "num_cuts", "penalty", "elapsed")


def _cell(column: str, value) -> str:
    _, width, formatter = _COLUMN_SPECS[column]
    text = "-" if value is None else formatter(value)
    return text.rjust(width)


def iteration_header(columns) -> str:
    return "  ".join(
        _COLUMN_SPECS[c][0].rjust(_COLUMN_SPECS[c][1])
        for c in _COLUMN_ORDER
        if c in columns
    )


def render_iteration(fields: dict) -> str:
    return "  ".join(_cell(c, fields[c]) for c in _COLUMN_ORDER if c in fields)


class BmLogger:
    """Logging for one bundle-method solver.

    Records go to ``pyodsp.alg.bm.<node_id>`` and carry the solver's node id
    and depth in ``record.pyodsp`` -- structured context a handler can use to
    place the line within the decomposition tree (the root flush left, each
    level below it indented) and to read the iteration numbers without parsing
    the row. ``pyodsp.configure_logging`` installs a handler that does exactly
    that; the library itself formats nothing.

    ``per_step`` drops this solver's progress and lifecycle lines to DEBUG.
    SDDP drives every stage node one bundle step at a time and resets it
    between visits, so its "completed / Total iterations: 1 / ..." lifecycle
    fires on every visit and is noise at INFO -- the lattice narrates instead.
    """

    def __init__(
        self,
        method: str,
        node_id: int | str,
        depth: int,
        level: int | None = logging.INFO,
    ) -> None:
        self.method = method
        self.node_id = node_id
        self.depth = depth
        self.per_step = False
        self._header_shown = False
        self.logger = logging.getLogger(f"pyodsp.alg.bm.{node_id}")
        if level is not None:
            self.logger.setLevel(level)

    def _extra(self, **more) -> dict:
        context = {"node_id": self.node_id, "depth": self.depth}
        context.update(more)
        return {"pyodsp": context}

    def _emit(self, message: str, extra: dict) -> None:
        level = logging.DEBUG if self.per_step else logging.INFO
        self.logger.log(level, message, extra=extra)

    def log_initialization(self, **kwargs) -> None:
        self._emit(f"Starting {self.method}", self._extra())
        for key, var in kwargs.items():
            self._emit(f"{key}: {var}", self._extra())

    def log_info(self, message: str) -> None:
        self._emit(message, self._extra())

    def log_debug(self, message: str) -> None:
        self.logger.debug(message, extra=self._extra())

    def log_iteration(self, **fields) -> None:
        if not self._header_shown:
            self._emit(iteration_header(fields), self._extra())
            self._header_shown = True
        self._emit(render_iteration(fields), self._extra(iteration=fields))

    def log_solution(self, solution) -> None:
        self.logger.debug(f"solution: {solution}", extra=self._extra())

    def log_status_optimal(self) -> None:
        self._emit(f"{self.method} terminated by optimality", self._extra())

    def log_status_max_iter(self) -> None:
        self._emit(
            f"{self.method} terminated by max iteration reached", self._extra()
        )

    def log_status_time_limit(self) -> None:
        self._emit(f"{self.method} terminated by time limit", self._extra())

    def log_infeasible(self) -> None:
        self._emit(f"{self.method} terminated by infeasibility", self._extra())

    def log_completion(self, iteration: int, objective_value: float | None) -> None:
        self._emit(f"{self.method} completed", self._extra())
        self._emit(f"Total iterations: {iteration}", self._extra())
        self._emit(f"Final objective value: {objective_value}", self._extra())
