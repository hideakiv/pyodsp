import logging


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

    def log_initialization(self, **kwargs):
        self.logger.info("Starting Bundle method")
        for key, var in kwargs.items():
            self.logger.info(f"{key}: {var}")

    def log_master_problem(self, iteration, objective_value, x):
        self.logger.info(f"Iteration {iteration}: {objective_value}")
        self.logger.debug(f"\tsolution: {x}")

    def log_sub_problem(self, idx, cut_type: str, coefficients, constant):
        self.logger.debug(f"\t{idx}\t{cut_type}\t{coefficients}\t{constant}")

    def log_status_optimal(self):
        self.logger.info("Bundle method terminated by optimality")

    def log_status_max_iter(self):
        self.logger.info("Bundle method terminated by max iteration reached")

    def log_completion(self, iteration, objective_value):
        self.logger.info("Bundle method completed")
        self.logger.info(f"Total iterations: {iteration}")
        self.logger.info(f"Final objective value: {objective_value}")
