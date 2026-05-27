from dataclasses import dataclass
import numpy as np

from balance import ScenarioParams


class Regime:
    def __init__(self, time: int):
        self.time = time
        pass

    def create_sample(self) -> ScenarioParams:
        pass


class NormalRegime(Regime):
    def __init__(self, time: int):
        super().__init__(time)

    def create_sample(self) -> ScenarioParams:
        sp = ScenarioParams(
            [100 for _ in range(self.time)],
            [100 for _ in range(self.time)],
        )
        return sp


class HotSunnyRegime(Regime):
    def __init__(self, time: int):
        super().__init__(time)

    def create_sample(self) -> ScenarioParams:
        sp = ScenarioParams(
            [200 for _ in range(self.time)],
            [200 for _ in range(self.time)],
        )
        return sp


class HotCloudyRegime(Regime):
    def __init__(self, time: int):
        super().__init__(time)

    def create_sample(self) -> ScenarioParams:
        sp = ScenarioParams(
            [300 for _ in range(self.time)],
            [300 for _ in range(self.time)],
        )
        return sp


class ColdSunnyRegime(Regime):
    def __init__(self, time: int):
        super().__init__(time)

    def create_sample(self) -> ScenarioParams:
        sp = ScenarioParams(
            [400 for _ in range(self.time)],
            [400 for _ in range(self.time)],
        )
        return sp


class ColdCloudyRegime(Regime):
    def __init__(self, time: int):
        super().__init__(time)

    def create_sample(self) -> ScenarioParams:
        sp = ScenarioParams(
            [500 for _ in range(self.time)],
            [500 for _ in range(self.time)],
        )
        return sp


@dataclass
class RegimeParams:
    regimes: list[Regime]
    transition_matrix: np.ndarray
