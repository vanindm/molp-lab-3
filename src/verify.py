"""Независимая проверка допустимости решения TSP (только NumPy, без объекта солвера).

Проверяется:
  1. целочисленность/бинарность x и симметрия тура;
  2. степени вершин: у каждой ровно одна входящая и одна исходящая дуга;
  3. отсутствие подциклов (один цикл длины n через вершину 0);
  4. пересчёт целевой функции по матрице расстояний.
"""

import numpy as np

TOL = 1e-7


def check_solution(dist, x):
    """x — полный вектор переменных формулировки MTZ (первые n*(n-1) — дуги).
    Возвращает словарь с результатами проверки."""
    n = len(dist)
    m = n * (n - 1)
    xv = np.asarray(x)[:m]
    report = {"n": n, "checks": [], "ok": True}

    def add(name, passed, detail=""):
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            report["ok"] = False

    # 1. бинарность
    add("x бинарны (0/1)", np.all((np.abs(xv) < TOL) | (np.abs(xv - 1) < TOL)),
        f"min={xv.min():.4f}, max={xv.max():.4f}")

    # 2. степени
    in_deg = np.zeros(n)
    out_deg = np.zeros(n)
    succ = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                k = i * (n - 1) + (j if j < i else j - 1)
                if xv[k] > 0.5:
                    out_deg[i] += 1
                    in_deg[j] += 1
                    succ[i] = j
    add("входящая степень = 1", np.allclose(in_deg, 1.0, atol=TOL), str(in_deg.astype(int)))
    add("исходящая степень = 1", np.allclose(out_deg, 1.0, atol=TOL), str(out_deg.astype(int)))

    # 3. один цикл длины n
    seen = set()
    v, cycle_len = 0, 0
    while v not in seen:
        seen.add(v)
        v = succ.get(v, -1)
        cycle_len += 1
        if v < 0:
            break
    add("один цикл длины n", v == 0 and cycle_len == n and len(seen) == n,
        f"длина цикла из 0: {cycle_len}, посещено: {len(seen)}")

    # 4. длина тура
    tour = [0]
    v = succ[0]
    while v != 0:
        tour.append(v)
        v = succ[v]
    length = float(sum(dist[tour[k], tour[k + 1]] for k in range(-1, n - 1)))
    report["tour"] = tour
    report["objective"] = length
    return report


def format_report(report):
    lines = [f"Независимая проверка (n={report['n']}), итог: "
             f"{'ДОПУСТИМО' if report['ok'] else 'НЕДОПУСТИМО'}"]
    for c in report["checks"]:
        lines.append(f"  [{'OK' if c['passed'] else 'FAIL'}] {c['name']}  {c['detail']}")
    if "tour" in report:
        lines.append(f"  тур: {' -> '.join(map(str, report['tour']))} -> 0")
        lines.append(f"  длина тура (пересчёт): {report['objective']:.2f}")
    return "\n".join(lines)
