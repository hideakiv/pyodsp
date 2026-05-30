from dataclasses import dataclass
import numpy as np

from balance import ConstantParams, ScenarioParams
from regime import RegimeParams


@dataclass
class BalanceParams:
    cp: ConstantParams
    sp_dict: dict[int, list[ScenarioParams]]
    tp_dict: dict[int, np.ndarray]
    init_level: float
    final_level: float
    final_penalty: float


def create_cp(num_stages: int, num_scenarios: int, time: int) -> ConstantParams:
    hours = np.arange(time) / 2.0
    proc_prices = [
        40.0 if h < 6 or h >= 22 else 55.0 if h < 12 else 70.0 if h < 18 else 62.0
        for h in hours
    ]
    max_proc = [
        45.0 if h < 6 or h >= 22 else 28.0 if h < 12 else 25.0 if h < 18 else 30.0
        for h in hours
    ]

    return ConstantParams(
        num_stages=num_stages,
        num_scenarios=num_scenarios,
        time=time,
        min_level=0.0,
        max_level=180.0,
        max_charge=22.0,
        max_discharge=22.0,
        charge_efficiency=0.92,
        max_purchase=40.0,
        max_sell=30.0,
        proc_prices=proc_prices,
        max_proc=max_proc,
        penalty=1e3,
    )


def create_static_scenarios(
    num_stages: int,
    num_scenarios: list[int],
    regime_params: RegimeParams,
    init_regime: int,
) -> BalanceParams:
    assert num_stages >= 2
    assert len(num_scenarios) == len(regime_params.regimes)

    init_level = 100.0
    final_level = 100.0
    final_penalty = 180.0
    time = 48
    total_scenarios = sum(num_scenarios)

    # create constant params
    cp = create_cp(num_stages, total_scenarios, time)

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
    for this_regime in range(len(num_scenarios)):
        for i in range(num_scenarios[this_regime]):
            s2 = 0
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

    return BalanceParams(cp, sp_dict, tp_dict, init_level, final_level, final_penalty)
