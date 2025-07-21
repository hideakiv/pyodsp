from dataclasses import dataclass
import random


@dataclass
class McspParams:
    K: int  # number of types of rolls
    N: list[int]  # number of rolls available (size: K)
    P: int  # number of pieces
    d: list[int]  # number of demand (size: P)
    L: list[int]  # length of a roll (size: K)
    c: list[float]  # cost of roll (size: K)
    l: list[int]  # length of piece (size: P)


def create_single() -> McspParams:
    return McspParams(
        K=1,
        N=[2000],
        P=5,
        d=[205, 2321, 143, 1089, 117],
        L=[110],
        c=[1.0],
        l=[70, 40, 55, 25, 35],
    )


def create_random(K: int, P: int, seed=42):
    """
    Create a random McspParams instance.
    - K: number of types of rolls
    - N: list[int], number of rolls available (size: K)
    - P: number of pieces (random, e.g., 3-8)
    - d: list[int], demand for each piece (size: P)
    - L: list[int], length of each roll (size: K)
    - c: list[float], cost of each roll (size: K)
    - l: list[int], length of each piece (size: P)
    """
    random.seed(seed)
    N = [random.randint(500, 3000) for _ in range(K)]
    d = [random.randint(50, 2500) for _ in range(P)]
    L = [random.randint(80, 200) for _ in range(K)]
    c = [round(random.uniform(0.5, 3.0), 2) for _ in range(K)]
    l = [random.randint(20, 80) for _ in range(P)]
    return McspParams(
        K=K,
        N=N,
        P=P,
        d=d,
        L=L,
        c=c,
        l=l,
    )
