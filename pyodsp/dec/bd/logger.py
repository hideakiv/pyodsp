import logging

from ..node._logger import ILogger


class BdLogger(ILogger):
    def __init__(self, level: int = logging.INFO):
        # Create a logger object
        self.logger = logging.getLogger("Benders Decomposition")
        self.logger.setLevel(level)

        # Create a console handler and set its level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create a formatter and set the format
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        ch.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(ch)

    def log_initialization(self, **kwargs):
        self.logger.info("Starting Benders decomposition")
        for key, var in kwargs.items():
            self.logger.info(f"{key}: {var}")

    def log_master_problem(self, iteration, objective_value, x):
        self.logger.info(f"Iteration {iteration}: {objective_value}")
        self.logger.debug(f"\t{x}")

    def log_sub_problem(self, idx, cut_type: str, coefficients, constant):
        self.logger.debug(f"\t{idx}\t{cut_type}\t{coefficients}\t{constant}")

    def log_finaliziation(self):
        self.logger.info("Finalizing Benders Decomposition")

    def log_completion(self, objective_value):
        self.logger.info("Benders decomposition completed")
        self.logger.info(f"Final objective value: {objective_value}")
