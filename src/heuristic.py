"""Жадная эвристика для TSP: ближайший непосещённый город + улучшение 2-opt.

Используется для сравнения с MILP: зазор по целевой функции и время работы.
"""

import time

import numpy as np


def nearest_neighbor(dist, start=0):
    n = len(dist)
    tour = [start]
    used = {start}
    while len(tour) < n:
        last = tour[-1]
        nxt = min((j for j in range(n) if j not in used), key=lambda j: dist[last, j])
        tour.append(nxt)
        used.add(nxt)
    return tour


def two_opt(tour, dist):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = tour[i], tour[(i + 1) % n]
                c_, d = tour[j], tour[(j + 1) % n]
                if dist[a, c_] + dist[b, d] < dist[a, b] + dist[c_, d] - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def tour_length(tour, dist):
    n = len(tour)
    return float(sum(dist[tour[k], tour[(k + 1) % n]] for k in range(n)))


def greedy_tsp(dist):
    """Полный жадный прогон: NN из всех стартов + 2-opt, лучший тур."""
    n = len(dist)
    t0 = time.perf_counter()
    best_tour, best_len = None, np.inf
    for s in range(n):
        t = two_opt(nearest_neighbor(dist, s), dist)
        L = tour_length(t, dist)
        if L < best_len:
            best_tour, best_len = t, L
    elapsed = time.perf_counter() - t0
    return best_tour, best_len, elapsed
