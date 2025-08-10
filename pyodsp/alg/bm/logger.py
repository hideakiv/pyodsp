import logging
from typing import List


class BmLogger:
    def __init__(
        self, method: str, node_id: int, depth: int, level: int = logging.INFO
    ) -> None:
        self.method = method
        self.node_id = node_id
        self.depth = depth
        # Create a logger object
        self.logger = logging.getLogger(f"{method} {node_id}")
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
        self.logger.info(f"Node: {self.node_id} - Starting {self.method}")
        for key, var in kwargs.items():
            self.logger.info(f"Node: {self.node_id} - {key}: {var}")

    def log_info(self, message: str) -> None:
        self.logger.info(f"Node: {self.node_id} - {message}")

    def log_debug(self, message: str) -> None:
        self.logger.debug(f"Node: {self.node_id} - {message}")

    def log_sub_problem(self, idx, cut_type: str, coefficients, constant) -> None:
        self.log_debug(f"\t{idx}\t{cut_type}\t{coefficients}\t{constant}")

    def log_status_optimal(self) -> None:
        self.log_info(f"{self.method} terminated by optimality")

    def log_status_max_iter(self) -> None:
        self.log_info(f"{self.method} terminated by max iteration reached")

    def log_status_time_limit(self) -> None:
        self.log_info(f"{self.method} terminated by time limit")

    def log_infeasible(self) -> None:
        self.log_info(f"{self.method} terminated by infeasibility")

    def log_completion(self, iteration: int, objective_value: float | None) -> None:
        self.log_info(f"{self.method} completed")
        self.log_info(f"Total iterations: {iteration}")
        self.log_info(f"Final objective value: {objective_value}")
