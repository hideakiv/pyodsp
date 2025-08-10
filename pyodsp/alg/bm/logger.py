import logging
from typing import List


class BmLogger:
    def __init__(self, node_id: int, depth: int, level: int = logging.INFO) -> None:
        self.node_id = node_id
        self.depth = depth
        # Create a logger object
        self.logger = logging.getLogger(f"Bundle Method {node_id}")
        self.logger.setLevel(level)

        # Create a console handler and set its level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create a formatter and set the format
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        ch.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(ch)

    def log_initialization(self, **kwargs) -> None:
        self.logger.info(f"Node: {self.node_id} - Starting Bundle method")
        for key, var in kwargs.items():
            self.logger.info(f"Node: {self.node_id} - {key}: {var}")

    def log_master_problem(
        self,
        iteration: int,
        lb: float | None,
        ub: float | None,
        x: List[float],
        numcuts: int,
        elapsed: float,
    ) -> None:
        if lb is None:
            lb = "-"
        else:
            lb = f"{lb:.4f}"
        if ub is None:
            ub = "-"
        else:
            ub = f"{ub:.4f}"
        self.logger.info(
            f"Node: {self.node_id} - Iteration: {iteration}\tLB: {lb}\t UB: {ub}\t NumCuts: {numcuts}\t Elapsed: {elapsed:.2f}"
        )
        self.logger.debug(f"Node: {self.node_id} - \tsolution: {x}")

    def log_sub_problem(self, idx, cut_type: str, coefficients, constant) -> None:
        self.logger.debug(
            f"Node: {self.node_id} - \t{idx}\t{cut_type}\t{coefficients}\t{constant}"
        )

    def log_status_optimal(self) -> None:
        self.logger.info(
            f"Node: {self.node_id} - Bundle method terminated by optimality"
        )

    def log_status_max_iter(self) -> None:
        self.logger.info(
            f"Node: {self.node_id} - Bundle method terminated by max iteration reached"
        )

    def log_status_time_limit(self) -> None:
        self.logger.info(
            f"Node: {self.node_id} - Bundle method terminated by time limit"
        )

    def log_infeasible(self) -> None:
        self.logger.info(
            f"Node: {self.node_id} - Bundle method terminated by infeasibility"
        )

    def log_completion(self, iteration: int, objective_value: float | None) -> None:
        self.logger.info(f"Node: {self.node_id} - Bundle method completed")
        self.logger.info(f"Node: {self.node_id} - Total iterations: {iteration}")
        self.logger.info(
            f"Node: {self.node_id} - Final objective value: {objective_value}"
        )
