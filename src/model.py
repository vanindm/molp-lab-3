"""MILP-модели TSP для ЛР-3.

Две эквивалентные формулировки устранения подциклов:
  1. MTZ (Miller–Tucker–Zemlin) — big-M связки;
  2. Однотоварный поток (Gavish–Graves) — «потоковая формулировка»
     из индивидуального варианта.

Обе решаются через scipy.optimize.milp (бэкенд HiGHS). С помощью
`relax=True` решается LP-релаксация (интегральность снята) — для
оценки интегрального зазора и сравнения силы формулировок.
"""

import os

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

# Индексация переменных для n вершин:
#   x[i,j], i!=j  ->  k = i*(n-1) + (j if j < i else j-1)      [n*(n-1) штук]
#   MTZ:  u[i], i=1..n-1  ->  n*(n-1) + (i-1)                  [n-1 штук]
#   Поток: f[i,j]           ->  n*(n-1) + n*(n-1)              [n*(n-1) штук]


def x_index(i, j, n):
    return i * (n - 1) + (j if j < i else j - 1)


def tour_from_x(x, n):
    """Восстанавливает тур (список вершин) из целочисленного решения x."""
    succ = {}
    for i in range(n):
        for j in range(n):
            if i != j and x[x_index(i, j, n)] > 0.5:
                succ[i] = j
    tour, v = [0], succ[0]
    while v != 0:
        tour.append(v)
        v = succ[v]
    return tour


def _degree_constraints(dist, ncols=None):
    n = len(dist)
    m = n * (n - 1)
    A = lil_matrix((2 * n, ncols if ncols is not None else m))
    lb = np.full(2 * n, 1.0)
    ub = np.full(2 * n, 1.0)
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, x_index(i, j, n)] = 1.0          # выход из i
                A[n + j, x_index(i, j, n)] = 1.0      # вход в j
    return A, lb, ub


def _objective(dist):
    n = len(dist)
    c = np.array([dist[i, j] for i in range(n) for j in range(n) if i != j])
    return c


def solve_mtz(dist, relax=False, time_limit=60.0, log_path=None, m_big=None):
    """MTZ: min sum d_ij x_ij; deg=1; u_i - u_j + n*x_ij <= n-1; 1 <= u <= n-1."""
    n = len(dist)
    m = n * (n -1)
    M = m_big if m_big is not None else n  # обоснованная big-M: M = n
    c = np.concatenate([_objective(dist), np.zeros(n - 1)])

    A, lb, ub = _degree_constraints(dist, ncols=m + n - 1)

    # MTZ-связки: для i!=j, i,j != 0: u_i - u_j + M*x_ij <= M-1
    A_sub = lil_matrix((n * n - 3 * n + 2, m + n - 1))
    r = 0
    lb_sub, ub_sub = [], []
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            A_sub[r, m + i - 1] = 1.0
            A_sub[r, m + j - 1] = -1.0
            A_sub[r, x_index(i, j, n)] = M
            lb_sub.append(-np.inf)
            ub_sub.append(M - 1)
            r += 1
    A = np.vstack([A.toarray(), A_sub.toarray()])
    lb = np.concatenate([lb, lb_sub])
    ub = np.concatenate([ub, ub_sub])

    integrality = np.zeros(m + n - 1)
    if not relax:
        integrality[:m] = 1  # x бинарные (0/1), u непрерывные
    bounds = Bounds(np.zeros(m + n - 1), np.concatenate([np.ones(m), np.full(n - 1, n - 1)]))
    bounds.lb[m:] = 1.0

    return _run(c, A, lb, ub, bounds, integrality, time_limit, log_path)


def solve_flow(dist, relax=False, time_limit=60.0, log_path=None):
    """Однотоварный поток: депо 0 поставляет n-1 единиц 'топлива';
    f_ij <= (n-1)*x_ij; баланс потока в каждой вершине."""
    n = len(dist)
    m = n * (n - 1)
    base = m            # сдвиг для f
    c = np.concatenate([_objective(dist), np.zeros(m)])

    A, lb, ub = _degree_constraints(dist, ncols=2 * m)

    rows = []
    lb2, ub2 = [], []
    # Баланс потока: sum_in f - sum_out f = b_i  (b_0 = n-1, иначе -1)
    B = lil_matrix((n, 2 * m))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            B[i, base + x_index(j, i, n)] += 1.0   # вход
            B[i, base + x_index(i, j, n)] -= 1.0   # выход
    rows.append(B.toarray())
    b = np.full(n, -1.0)
    b[0] = n - 1
    lb2 += b.tolist()
    ub2 += b.tolist()

    # Связка f_ij <= (n-1) x_ij
    F = lil_matrix((m, 2 * m))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            k = x_index(i, j, n)
            F[k, base + k] = 1.0
            F[k, k] = -(n - 1)
    rows.append(F.toarray())
    lb2 += np.full(m, -np.inf).tolist()
    ub2 += np.zeros(m).tolist()

    A = np.vstack([A.toarray()] + rows)
    lb = np.concatenate([lb, lb2])
    ub = np.concatenate([ub, ub2])

    integrality = np.zeros(2 * m)
    if not relax:
        integrality[:m] = 1
    bounds = Bounds(np.zeros(2 * m), np.concatenate([np.ones(m), np.full(m, n - 1)]))

    return _run(c, A, lb, ub, bounds, integrality, time_limit, log_path)


def _run(c, A, lb, ub, bounds, integrality, time_limit, log_path):
    import time
    constraints = LinearConstraint(A, lb, ub)

    # HiGHS пишет лог через файловый дескриптор stdout (C++), поэтому
    # перенаправляем fd 1 в файл лога на время решения.
    saved_fd = os.dup(1)
    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    if log_file:
        os.dup2(log_file.fileno(), 1)
    t0 = time.perf_counter()
    try:
        res = milp(
            c=c, constraints=constraints, bounds=bounds,
            integrality=integrality,
            options={"time_limit": time_limit, "disp": True,
                     "mip_rel_gap": 0.0},
        )
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        if log_file:
            log_file.close()
    elapsed = time.perf_counter() - t0
    return res, elapsed
