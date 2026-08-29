import logging

INDENT = "  "

# The iteration row, built once from named fields. bm, pbm and rbm all
# reported the same line -- iteration, bound, incumbent, cut count, elapsed --
# differing only in whether a center bound (CB) and a penalty (u) column were
# present. Columns are emitted in this order, and only when the caller passes
# them; a None value renders as "-".
_COLUMN_ORDER = ("iteration", "lb", "center_bound", "ub", "num_cuts", "penalty", "elapsed")
_COLUMN_LABELS = {
    "iteration": "Iteration",
    "lb": "LB",
    "center_bound": "CB",
    "ub": "UB",
    "num_cuts": "NumCuts",
    "penalty": "u",
    "elapsed": "Elapsed",
}


def _render_cell(column: str, value) -> str:
    if value is None:
        return "-"
    if column == "elapsed":
        return f"{value:.2f}"
    if column in ("iteration", "num_cuts", "penalty"):
        return f"{value}"
    return f"{value:.4f}"


def render_iteration(fields: dict) -> str:
    return "\t".join(
        f"{_COLUMN_LABELS[c]}: {_render_cell(c, fields[c])}"
        for c in _COLUMN_ORDER
        if c in fields
    )


class BmLogger:
    """Logging for one bundle-method solver.

    Records go to ``pyodsp.alg.bm.<node_id>`` and carry the solver's node id
    and depth in ``record.pyodsp`` -- structured context a handler can use to
    place the line within the decomposition tree (the root flush left, each
    level below it indented) and to read the iteration numbers without parsing
    the row. ``pyodsp.configure_logging`` installs a handler that does exactly
    that; the library itself formats nothing.
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
        self.logger = logging.getLogger(f"pyodsp.alg.bm.{node_id}")
        if level is not None:
            self.logger.setLevel(level)

    def _extra(self, **more) -> dict:
        context = {"node_id": self.node_id, "depth": self.depth}
        context.update(more)
        return {"pyodsp": context}

    def log_initialization(self, **kwargs) -> None:
        self.logger.info(f"Starting {self.method}", extra=self._extra())
        for key, var in kwargs.items():
            self.logger.info(f"{key}: {var}", extra=self._extra())

    def log_info(self, message: str) -> None:
        self.logger.info(message, extra=self._extra())

    def log_debug(self, message: str) -> None:
        self.logger.debug(message, extra=self._extra())

    def log_iteration(self, **fields) -> None:
        self.logger.info(
            render_iteration(fields), extra=self._extra(iteration=fields)
        )

    def log_solution(self, solution) -> None:
        self.logger.debug(f"solution: {solution}", extra=self._extra())

    def log_status_optimal(self) -> None:
        self.logger.info(
            f"{self.method} terminated by optimality", extra=self._extra()
        )

    def log_status_max_iter(self) -> None:
        self.logger.info(
            f"{self.method} terminated by max iteration reached", extra=self._extra()
        )

    def log_status_time_limit(self) -> None:
        self.logger.info(
            f"{self.method} terminated by time limit", extra=self._extra()
        )

    def log_infeasible(self) -> None:
        self.logger.info(
            f"{self.method} terminated by infeasibility", extra=self._extra()
        )

    def log_completion(self, iteration: int, objective_value: float | None) -> None:
        self.logger.info(f"{self.method} completed", extra=self._extra())
        self.logger.info(f"Total iterations: {iteration}", extra=self._extra())
        self.logger.info(
            f"Final objective value: {objective_value}", extra=self._extra()
        )
