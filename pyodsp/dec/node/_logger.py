import logging
from abc import ABC, abstractmethod
from typing import Any


class ILogger(ABC):
    @abstractmethod
    def log_info(self, text: str):
        pass

    @abstractmethod
    def log_debug(self, text: str):
        pass

    @abstractmethod
    def log_initialization(self, **kwargs):
        pass

    @abstractmethod
    def log_sub_problem(self, idx, cut_type: str, coefficients: Any, constant: float):
        pass

    @abstractmethod
    def log_completion(
        self, objective_value: float | None, *, label: str = "Final objective value"
    ):
        pass


class AlgLogger(ILogger):
    """The default ILogger for the decomposition algorithms.

    Every algorithm used to report the same handful of lines through its own
    near-identical logger class. This is the single implementation, told which
    algorithm it speaks for: ``method`` names it in the messages and ``channel``
    picks the ``pyodsp.dec.<channel>`` logger the records go to. The library
    installs no handler on that logger -- see ``pyodsp.configure_logging``. A
    ``level`` of ``None`` leaves the logger's level alone so it inherits.
    """

    def __init__(self, method: str, channel: str, level: int | None = logging.INFO):
        self.method = method
        self.logger = logging.getLogger(f"pyodsp.dec.{channel}")
        if level is not None:
            self.logger.setLevel(level)

    def log_info(self, text: str) -> None:
        self.logger.info(text)

    def log_debug(self, text: str) -> None:
        self.logger.debug(text)

    def log_initialization(self, **kwargs) -> None:
        self.logger.info(f"Starting {self.method}")
        for key, var in kwargs.items():
            self.logger.info(f"{key}: {var}")

    def log_sub_problem(
        self, idx, cut_type: str, coefficients: Any, constant: float
    ) -> None:
        self.logger.debug(f"\t{idx}\t{cut_type}\t{coefficients}\t{constant}")

    def log_completion(
        self, objective_value: float | None, *, label: str = "Final objective value"
    ) -> None:
        if objective_value is None:
            self.logger.info(f"{self.method} completed")
        else:
            self.logger.info(
                f"{self.method} completed. {label}: {objective_value}"
            )
