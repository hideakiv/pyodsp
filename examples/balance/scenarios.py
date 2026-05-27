from dataclasses import dataclass
import numpy as np

from balance import ConstantParams, ScenarioParams
from regime import RegimeParams


@dataclass
class BalanceParams:
    cp: ConstantParams
    sp_dict: dict[int, list[ScenarioParams]]
    tp_dict: dict[int, np.ndarray]


def create_cp() -> ConstantParams:
    pass


def create_static_scenarios(
    num_stages: int,
    num_scenarios: list[int],
    regime_params: RegimeParams,
    init_regime: int,
) -> BalanceParams:
    assert num_stages >= 2
    assert len(num_scenarios) == len(regime_params.regimes)

    # create constant params
    cp = create_cp()

    # create list of scenarios
    scenarios: list[ScenarioParams] = []
    for regime_id, regime in enumerate(regime_params.regimes):
        for _ in range(num_scenarios[regime_id]):
            schedule = regime.create_sample()
            scenarios.append(schedule)
    # use scenarios as states for all stages
    sp_dict = {}
    for stage in range(1, num_stages):
        sp_dict[stage] = scenarios.copy()

    # create initial transition probability
    total_scenarios = sum(num_scenarios)
    init_tp = np.zeros((1, total_scenarios))
    s2 = 0
    for next_regime in range(len(num_scenarios)):
        for j in range(num_scenarios[next_regime]):
            raw_prob = regime_params.transition_matrix[init_regime, next_regime]
            init_tp[0, s2] = raw_prob / num_scenarios[next_regime]
            s2 += 1

    # create regular transition probability
    tp = np.zeros((total_scenarios, total_scenarios))
    s1 = 0
    s2 = 0
    for this_regime in range(len(num_scenarios)):
        for i in range(num_scenarios[this_regime]):
            for next_regime in range(len(num_scenarios)):
                for j in range(num_scenarios[next_regime]):
                    raw_prob = regime_params.transition_matrix[this_regime, next_regime]
                    tp[s1, s2] = raw_prob / num_scenarios[next_regime]
                    s2 += 1
            s1 += 1

    # use tp as transition probability for all stages except 0
    tp_dict = {}
    tp_dict[0] = init_tp
    for stage in range(1, num_stages):
        tp_dict[stage] = tp

    return BalanceParams(cp, sp_dict, tp_dict)
