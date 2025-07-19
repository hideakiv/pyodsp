from dataclasses import dataclass
import random


@dataclass
class UcParams:
    u0: int  # state at time 0. 0: off, 1: on.
    p0: float  # power at time 0.
    last_up: int  # last time unit was up or 0 if u0 == 1
    last_dn: int  # last time unit was dn or 0 if u0 == 0

    UT: int  # minimum up time.
    DT: int  # minimum down time.

    P_up: float  # maximum power.
    P_dn: float  # minimum power.

    RU: float  # ramp up. Less than P_up - P_dn.
    SU: float  # start up. Less than P_up.
    RD: float  # ramp down. Less than P_up - P_dn.
    SD: float  # shut down. Less than P_up.

    cp: list[float]  # cost of power generation. size: num_seg. strictly increasing.
    lp: list[
        float
    ]  # segment upper bound. size: num_seg. last element is equal to P_up. strictly increasing.
    c_hot: float  # cost of hot start up.
    c_cold: float  # cost of cold start up.
    DT_cold: int  # must be greater than DT.
    c_run: float  # cost of running.
    c_shut: float  # cost of shut down.


def create_params(num_seg: int, seed: int) -> UcParams:
    random.seed(seed)
    # State at time 0: 0 (off) or 1 (on)
    u0 = random.randint(0, 1)

    # Minimum up/down times (2-6)
    UT = random.randint(2, 6)
    DT = random.randint(2, 6)
    # DT_cold > DT
    DT_cold = DT + random.randint(1, 5)
    # Initial up/down times: InitU in [0, UT] if u0==1, else 0; InitD in [0, DT] if u0==0, else 0
    if u0 == 0:
        last_up = -random.randint(1, DT * 2)
        last_dn = 0
    else:
        last_up = 0
        last_dn = -random.randint(1, UT * 2)
    # Power limits
    P_dn = round(random.uniform(10, 30), 2)
    P_up = round(P_dn + random.uniform(20, 70), 2)

    # Power at time 0: if off, 0; if on, between P_dn and P_up
    if u0 == 0:
        p0 = 0.0
    else:
        p0 = round(random.uniform(P_dn, P_up), 2)
    # Ramp up/down (less than P_up - P_dn)
    ramp_limit = P_up - P_dn
    RU = round(random.uniform(ramp_limit / 3, ramp_limit), 2)
    RD = round(random.uniform(ramp_limit / 3, ramp_limit), 2)
    # Start up/shut down (less than P_up)
    SU = round(random.uniform((P_up - 1) / 2, P_up - 1), 2)
    SD = round(random.uniform((P_up - 1) / 2, P_up - 1), 2)
    # Cost segments (strictly increasing)
    cp = []
    last_cp = random.uniform(10, 30)
    for _ in range(num_seg):
        last_cp += random.uniform(5, 20)
        cp.append(round(last_cp, 2))
    # Segment upper bounds (strictly increasing, last = P_up)
    lp = []
    for i in range(num_seg - 1):
        lp.append(round(P_dn + (P_up - P_dn) * (i + 1) / num_seg, 2))
    lp.append(P_up)
    # Startup costs
    c_hot = round(random.uniform(10, 100), 2)
    c_cold = round(c_hot + random.uniform(10, 100), 2)
    # Running/shutdown costs
    c_run = round(random.uniform(5, 20), 2)
    c_shut = round(random.uniform(5, 20), 2)
    return UcParams(
        u0=u0,
        p0=p0,
        last_up=last_up,
        last_dn=last_dn,
        UT=UT,
        DT=DT,
        P_up=P_up,
        P_dn=P_dn,
        RU=RU,
        SU=SU,
        RD=RD,
        SD=SD,
        cp=cp,
        lp=lp,
        c_hot=c_hot,
        c_cold=c_cold,
        DT_cold=DT_cold,
        c_run=c_run,
        c_shut=c_shut,
    )


def create_params_dict(num_gens: int, num_seg: int, seed) -> dict[int, UcParams]:
    params = {}

    for k in range(1, num_gens + 1):
        params[k] = create_params(num_seg, seed + k)

    return params


def create_demand(
    num_day: int, params: dict[int, UcParams], seed
) -> tuple[int, list[float]]:
    num_time = 48 * num_day
    random.seed(seed)
    # Simulate weather: 0 = clear (more solar), 1 = cloudy (less solar)
    weather = random.choices([0, 1], weights=[0.6, 0.4], k=num_day)
    # Get total max power from all generators
    total_pmax = sum(p.P_up for p in params.values()) * 0.8
    demand = []
    for d in range(num_day):
        for t in range(48):
            # --- Initial time slots: consider initial state and ramping ---
            if d == 0 and t < 4:
                # Estimate initial available power from units that are ON
                p_init = sum(p.p0 for p in params.values() if p.u0 == 1)
                # Estimate max ramp up from all units
                ramp_up = sum(p.RU for p in params.values())
                # For t=0, demand cannot exceed p_init + small noise
                if t == 0:
                    base = p_init + random.uniform(-0.01, 0.01) * total_pmax
                else:
                    # For t>0, demand cannot exceed p_init + t * ramp_up
                    base = p_init + t * ramp_up
                    # But also respect the time-of-day profile
                    if t < 12:
                        base = min(base, random.uniform(0.35, 0.5) * total_pmax)
                    elif t < 20:
                        base = min(base, random.uniform(0.5, 0.7) * total_pmax)
                    else:
                        base = min(base, random.uniform(0.7, 0.9) * total_pmax)
                base = max(base, 0.0)
            else:
                # Night: 0-12, 36-47 (0:00-6:00, 18:00-24:00)
                if t < 12 or t >= 36:
                    base = random.uniform(0.35, 0.5) * total_pmax
                # Morning ramp: 12-20 (6:00-10:00)
                elif t < 20:
                    base = random.uniform(0.5, 0.7) * total_pmax
                # Day: 20-36 (10:00-18:00)
                else:
                    base = random.uniform(0.7, 0.9) * total_pmax
                # Solar effect: reduce load during day if clear weather
                if 20 <= t < 36 and weather[d] == 0:
                    # Solar reduces load by 10-25%
                    base *= random.uniform(0.75, 0.9)
                # Add some random noise
                base += random.uniform(-0.03, 0.03) * total_pmax
            demand.append(round(max(base, 0.0), 2))
    return num_time, demand


def create_random(
    num_day: int, num_gens: int, num_seg: int, seed=42
) -> tuple[int, list[float], dict[int, UcParams]]:
    params = create_params_dict(num_gens, num_seg, seed)
    num_time, demand = create_demand(num_day, params, seed)

    return num_time, demand, params
