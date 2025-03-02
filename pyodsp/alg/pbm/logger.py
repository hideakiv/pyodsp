import logging
from typing import List


class PbmLogger:
    class ContextFilter(logging.Filter):
        def __init__(self, node_id, depth):
            super().__init__()
            self.node_id = node_id
            self.depth = depth

        def filter(self, record):
            record.node_id = self.node_id
            record.depth = self.depth
            return True

    def __init__(self, node_id: int, depth: int) -> None:
        self.node_id = node_id
        self.depth = depth
        # Create a logger object
        self.logger = logging.getLogger("Regularized Bundle Method")
        self.logger.setLevel(logging.INFO)

        # Create a console handler and set its level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create a formatter and set the format
        formatter = logging.Formatter("%(levelname)s - Node: %(node_id)s - %(message)s")
        ch.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(ch)

        # Add context filter to the logger
        context_filter = self.ContextFilter(node_id, depth)
        self.logger.addFilter(context_filter)

    def log_initialization(self, **kwargs) -> None:
        self.logger.info("Starting Regularized Bundle method")
        for key, var in kwargs.items():
            self.logger.info(f"{key}: {var}")

    def log_master_problem(
        self,
        iteration: int,
        lb: float | None,
        cb: float | None,
        ub: float | None,
        x: List[float],
        numcuts: int,
        elapsed: float
    ) -> None:
        if lb is None:
            lb = "-"
        else:
            lb = f"{lb:.4f}"
        if cb is None:
            cb = "-"
        else:
            cb = f"{cb:.4f}"
        if ub is None:
            ub = "-"
        else:
            ub = f"{ub:.4f}"
        self.logger.info(f"Iteration: {iteration}\tLB: {lb}\t CB: {cb}\t UB: {ub}\t NumCuts: {numcuts}\t Elapsed: {elapsed:.2f}")
        self.logger.debug(f"\tsolution: {x}")

    def log_sub_problem(self, idx, cut_type: str, coefficients, constant) -> None:
        self.logger.debug(f"\t{idx}\t{cut_type}\t{coefficients}\t{constant}")

    def log_status_optimal(self) -> None:
        self.logger.info("Regularized Bundle method terminated by optimality")

    def log_status_max_iter(self) -> None:
        self.logger.info(
            "Regularized Bundle method terminated by max iteration reached"
        )

    def log_status_time_limit(self) -> None:
        self.logger.info("Regularized Bundle method terminated by time limit")

    def log_completion(self, iteration: int, objective_value: float | None) -> None:
        self.logger.info("Regularized Bundle method completed")
        self.logger.info(f"Total iterations: {iteration}")
        self.logger.info(f"Final objective value: {objective_value}")
