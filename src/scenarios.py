"""Сценарии для TSP (вариант 9, n=12).

Предметная трактовка: курьер обслуживает 12 точек из депо 0.
  - base    : базовая матрица расстояний;
  - deficit : «дорожает топливо в северной части города» —
              все дуги, обе концы которых севернее y=60, дорожают на 40%;
  - stress  : двойное ухудшение — север дорожает уже в 2 раза и
              одновременно перекрываются две самые длинные дуги
              оптимального базового тура (два ДТП/ремонта дорог).
"""

import numpy as np

NORTH_LINE = 60.0
DEFICIT_FACTOR = 1.4


def base_scenario(dist):
    return dist.copy(), "базовая матрица расстояний"


def deficit_scenario(dist, coords):
    d = dist.copy()
    north = coords[:, 1] > NORTH_LINE
    mask = np.outer(north, north)
    d = np.where(mask, np.round(d * DEFICIT_FACTOR, 2), d)
    np.fill_diagonal(d, 0.0)
    return d, (f"севернее y={NORTH_LINE:.0f} дуги дороже "
               f"в {DEFICIT_FACTOR} раз ({int(mask.sum())} дуг)")


def stress_scenario(dist, coords, base_tour):
    d, desc = deficit_scenario(dist, coords)
    n = len(d)
    # стресс = удорожание в 2 раза (вместо 1.4) + перекрытие двух
    # самых длинных дуг оптимального базового тура
    north = coords[:, 1] > NORTH_LINE
    mask = np.outer(north, north)
    d = np.where(mask, np.round(dist * 2.0, 2), d)
    np.fill_diagonal(d, 0.0)
    legs = [(base_tour[k], base_tour[(k + 1) % n]) for k in range(n)]
    legs.sort(key=lambda e: -dist[e[0], e[1]])
    blocked = legs[:2]
    for a, b in blocked:
        d[a, b] = d[b, a] = 1e6
    txt = "; ".join(f"{a}->{b}" for a, b in blocked)
    return d, (f"север дороже в 2 раза ({int(mask.sum())} дуг); "
               f"перекрыты дуги: {txt}")


SCENARIOS = ["base", "deficit", "stress"]


def build(name, dist, coords, base_tour=None):
    if name == "base":
        return base_scenario(dist)
    if name == "deficit":
        return deficit_scenario(dist, coords)
    if name == "stress":
        return stress_scenario(dist, coords, base_tour)
    raise ValueError(name)
