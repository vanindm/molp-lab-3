"""Генерация данных варианта ЛР-3 (V=9, TSP, 12 вершин).

Seed = 10000 + 100*Q + N = 10000 + 100*2 + 4 = 10204 (N=4, Q=2, Y=26).

Вершины — случайные точки на плоскости [0, 100] x [0, 100]
(генератор numpy.random.default_rng(seed)), матрица расстояний —
округлённые евклидовы расстояния (симметричная, нулевая диагональ).
Сценарии модифицируют эту базовую матрицу (см. scenarios.py).
"""

import json
import os

import numpy as np

SEED = 10204
N_POINTS = 12
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate(seed=SEED, n_points=N_POINTS):
    """Возвращает (coords, dist): координаты и матрица расстояний n x n."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(n_points, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    dist = np.round(dist, 2)
    np.fill_diagonal(dist, 0.0)
    return coords, dist


def save_variant(seed=SEED, n_points=N_POINTS, path=None):
    coords, dist = generate(seed, n_points)
    if path is None:
        path = os.path.join(BASE_DIR, "data", f"tsp_{n_points}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "seed": seed,
        "n_points": n_points,
        "coords": [[round(float(x), 4), round(float(y), 4)] for x, y in coords],
        "dist": dist.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def load_variant(n_points=N_POINTS):
    path = os.path.join(BASE_DIR, "data", f"tsp_{n_points}.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return np.array(payload["coords"]), np.array(payload["dist"])


if __name__ == "__main__":
    p = save_variant()
    print(f"saved: {p}")
