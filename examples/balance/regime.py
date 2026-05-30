from dataclasses import dataclass
from typing import Optional
import numpy as np

from balance import ScenarioParams


class Regime:
    def __init__(self, time: int, seed: Optional[int] = None):
        self.time = time
        self.rng = np.random.default_rng(seed)
        self.nominal_demand = self._nominal_demand()
        self.nominal_price = self._nominal_price()

    def _daily_profile(self, values: list[float]) -> np.ndarray:
        hours = np.arange(self.time) / 2.0
        profile_hours = [0, 6, 10, 16, 20, 24]
        return np.interp(hours, profile_hours, values)

    def _nominal_demand(self) -> np.ndarray:
        raise NotImplementedError

    def _nominal_price(self) -> np.ndarray:
        raise NotImplementedError

    def create_sample(self) -> ScenarioParams:
        demand = self.nominal_demand.copy()
        price = self.nominal_price.copy()

        for t in range(self.time):
            end_of_day = t < 6 or t >= self.time - 6
            noise_scale = 0.03 if end_of_day else 0.12
            demand[t] = max(
                0.0, demand[t] + self.rng.normal(0.0, noise_scale * demand[t])
            )

        for t in range(self.time):
            conditional_shift = 0.15 * (demand[t] - self.nominal_demand[t])
            price_noise = self.rng.normal(0.0, 0.05 * self.nominal_price[t])
            price[t] = max(0.0, self.nominal_price[t] + conditional_shift + price_noise)

        return ScenarioParams(demand.tolist(), price.tolist())


class NormalRegime(Regime):
    def __init__(self, time: int, seed: int | None = None):
        super().__init__(time, seed)

    def _nominal_demand(self) -> np.ndarray:
        return self._daily_profile([60, 75, 95, 90, 70, 60])

    def _nominal_price(self) -> np.ndarray:
        return self._daily_profile([35, 50, 70, 60, 45, 35])


class HotSunnyRegime(Regime):
    def __init__(self, time: int, seed: int | None = None):
        super().__init__(time, seed)

    def _nominal_demand(self) -> np.ndarray:
        return self._daily_profile([70, 100, 155, 130, 95, 70])

    def _nominal_price(self) -> np.ndarray:
        return self._daily_profile([40, 65, 105, 85, 55, 40])


class HotCloudyRegime(Regime):
    def __init__(self, time: int, seed: int | None = None):
        super().__init__(time, seed)

    def _nominal_demand(self) -> np.ndarray:
        return self._daily_profile([75, 105, 145, 135, 100, 75])

    def _nominal_price(self) -> np.ndarray:
        return self._daily_profile([42, 68, 100, 88, 58, 42])


class ColdSunnyRegime(Regime):
    def __init__(self, time: int, seed: int | None = None):
        super().__init__(time, seed)

    def _nominal_demand(self) -> np.ndarray:
        return self._daily_profile([80, 120, 140, 125, 90, 80])

    def _nominal_price(self) -> np.ndarray:
        return self._daily_profile([45, 78, 95, 82, 55, 45])


class ColdCloudyRegime(Regime):
    def __init__(self, time: int, seed: int | None = None):
        super().__init__(time, seed)

    def _nominal_demand(self) -> np.ndarray:
        return self._daily_profile([85, 130, 155, 140, 100, 85])

    def _nominal_price(self) -> np.ndarray:
        return self._daily_profile([50, 85, 110, 95, 60, 50])


@dataclass
class RegimeParams:
    regimes: list[Regime]
    transition_matrix: np.ndarray
