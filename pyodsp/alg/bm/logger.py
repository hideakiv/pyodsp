import logging
from typing import List


class BmLogger:
    def __init__(self) -> None:
        # Create a logger object
        self.logger = logging.getLogger("Bundle Method")
        self.logger.setLevel(logging.DEBUG)

        # Create a console handler and set its level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create a formatter and set the format
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        ch.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(ch)

    def log_initialization(self, **kwargs) -> None:
        self.logger.info("Starting Bundle method")
        for key, var in kwargs.items():
            self.logger.info(f"{key}: {var}")

    def log_master_problem(
        self, iteration: int, lb: float | None, ub: float | None, x: List[float]
    ) -> None:
        if lb is None:
            lb = "-"
        if ub is None:
            ub = "-"
        self.logger.info(f"Iteration {iteration}:\tLB: {lb}, UB: {ub}")
        self.logger.debug(f"\tsolution: {x}")

    def log_sub_problem(self, idx, cut_type: str, coefficients, constant) -> None:
        self.logger.debug(f"\t{idx}\t{cut_type}\t{coefficients}\t{constant}")

    def log_status_optimal(self) -> None:
        self.logger.info("Bundle method terminated by optimality")

    def log_status_max_iter(self) -> None:
        self.logger.info("Bundle method terminated by max iteration reached")

    def log_completion(self, iteration: int, objective_value: float | None) -> None:
        self.logger.info("Bundle method completed")
        self.logger.info(f"Total iterations: {iteration}")
        self.logger.info(f"Final objective value: {objective_value}")
