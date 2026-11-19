"""Главный скрипт ЛР-3 (вариант 9: TSP, потоковая формулировка, n=12).

Одна команда запускает полный цикл:
  генерация данных -> MILP/LP-релаксация (MTZ и потоковая формулировки)
  -> независимая проверка -> жадная эвристика -> сценарии
  -> анализ big-M -> кривая масштаба -> логи, CSV, графики.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.generate_data import save_variant, load_variant, SEED, N_POINTS
from src.model import solve_mtz, solve_flow, tour_from_x
from src.verify import check_solution, format_report
from src.heuristic import greedy_tsp
from src.scenarios import SCENARIOS, build
from src.scaling import scaling_study

LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")
RES_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
for d in (LOG_DIR, RES_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)


def status_text(res):
    if res is None or not res.success:
        return "infeasible/ошибка"
    return "optimal (HiGHS)"


def plot_tour(coords, tour, dist, title, path):
    n = len(tour)
    fig, ax = plt.subplots(figsize=(6, 6))
    loop = tour + [tour[0]]
    xs = [coords[v, 0] for v in loop]
    ys = [coords[v, 1] for v in loop]
    ax.plot(xs, ys, "b-o", lw=1.5, ms=6, zorder=2)
    for k in range(n):
        a, b = loop[k], loop[k + 1]
        ax.annotate("", xy=(coords[b, 0], coords[b, 1]),
                    xytext=(coords[a, 0], coords[a, 1]),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    for v in range(len(coords)):
        ax.annotate(str(v), (coords[v, 0], coords[v, 1]),
                    textcoords="offset points", xytext=(6, 6), fontsize=10)
    L = sum(dist[loop[k], loop[k + 1]] for k in range(n))
    ax.set_title(f"{title}\nдлина тура = {L:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    print(f"=== ЛР-3, вариант 9 (TSP, потоковая формулировка), seed={SEED}, n={N_POINTS} ===\n")

    data_path = save_variant()
    coords, dist = load_variant()
    n = len(dist)
    print(f"данные: {data_path}\n")

    scenario_rows = []
    verify_texts = []
    lp_rows = []
    base_tour = None

    for sc in SCENARIOS:
        d, desc = build(sc, dist, coords, base_tour)
        print(f"--- сценарий {sc}: {desc}")

        res_milp, t_milp = solve_mtz(
            d, relax=False, time_limit=120.0,
            log_path=os.path.join(LOG_DIR, f"highs_milp_{sc}.log"))
        res_lp_mtz, t_lp_mtz = solve_mtz(d, relax=True, time_limit=120.0)
        res_lp_flow, t_lp_flow = solve_flow(d, relax=True, time_limit=120.0)

        assert res_milp.success, f"сценарий {sc}: статус {status_text(res_milp)}"
        x = res_milp.x
        tour = tour_from_x(x, n)
        if sc == "base":
            base_tour = tour

        rep = check_solution(d, x)
        verify_texts.append(f"[сценарий {sc}]\n" + format_report(rep))

        h_tour, h_len, h_time = greedy_tsp(d)
        gap_h = (h_len - rep["objective"]) / rep["objective"]

        lp_best = min(res_lp_mtz.fun, res_lp_flow.fun)
        int_gap = (rep["objective"] - lp_best) / rep["objective"]
        frac = float(np.sum(np.abs(x[: n * (n - 1)] - np.round(x[: n * (n - 1)])) > 1e-6))

        scenario_rows.append({
            "scenario": sc, "description": desc,
            "status": status_text(res_milp),
            "objective": round(rep["objective"], 2),
            "tour": "->".join(map(str, tour + [0])),
            "milp_time_s": round(t_milp, 4),
            "lp_relax_obj": round(lp_best, 4),
            "lp_mtz_obj": round(res_lp_mtz.fun, 4),
            "lp_flow_obj": round(res_lp_flow.fun, 4),
            "integrality_gap": round(int_gap, 6),
            "fractional_arcs_in_lp": frac,
            "greedy_obj": round(h_len, 2),
            "greedy_time_s": round(h_time, 4),
            "greedy_gap": round(gap_h, 6),
        })
        lp_rows.append({"scenario": sc, "formulation": "MTZ", "lp_obj": round(res_lp_mtz.fun, 4)})
        lp_rows.append({"scenario": sc, "formulation": "flow", "lp_obj": round(res_lp_flow.fun, 4)})

        plot_tour(coords, tour, d, f"Сценарий «{sc}»", os.path.join(FIG_DIR, f"tour_{sc}.png"))
        print(f"    MILP: {rep['objective']:.2f} ({t_milp:.3f} с), "
              f"LP: {lp_best:.2f}, gap: {int_gap:.2%}, жадный: {h_len:.2f} (gap {gap_h:.2%})")

    df_sc = pd.DataFrame(scenario_rows)
    df_sc.to_csv(os.path.join(RES_DIR, "scenarios.csv"), index=False)
    pd.DataFrame(lp_rows).to_csv(os.path.join(RES_DIR, "lp_relaxations.csv"), index=False)
    with open(os.path.join(RES_DIR, "verification.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(verify_texts) + "\n")

    # --- big-M: обоснованная M=n против завышенной M=1000 ---
    # (на n=12 HiGHS решает в корне, поэтому эксперимент ставится на n=40,
    #  где ветвление реально происходит)
    print("\n--- анализ big-M (MTZ, масштаб n=40)")
    from src.generate_data import generate
    _, dist40 = generate(n_points=40)
    bigm_rows = []
    for M in (40, 1000):
        res, t = solve_mtz(dist40, relax=False, time_limit=180.0, m_big=M,
                              log_path=os.path.join(LOG_DIR, f"highs_bigm_{M}.log"))
        bigm_rows.append({
            "n": 40, "M": M,
            "objective": round(res.fun, 2) if res.success else None,
            "time_s": round(t, 4),
            "nodes": int(getattr(res, "mip_node_count", -1)),
        })
        print(f"    M={M}: obj={res.fun:.2f}, {t:.3f} с, узлов={bigm_rows[-1]['nodes']}")
    pd.DataFrame(bigm_rows).to_csv(os.path.join(RES_DIR, "big_m.csv"), index=False)

    # --- кривая масштаба ---
    print("\n--- кривая масштаба (10, 12, 20, 40 вершин)")
    sc_rows = scaling_study()
    df_scale = pd.DataFrame(sc_rows)
    df_scale.to_csv(os.path.join(RES_DIR, "scaling.csv"), index=False)
    print(df_scale.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(df_scale["n"], df_scale["milp_time_s"], "o-", label="MILP")
    axes[0].plot(df_scale["n"], df_scale["lp_mtz_time_s"], "s--", label="LP-релаксация (MTZ)")
    axes[0].set_xlabel("число вершин n"); axes[0].set_ylabel("время, с")
    axes[0].set_yscale("log"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(df_scale["n"], df_scale["integrality_gap"] * 100, "o-", color="darkred")
    axes[1].set_xlabel("число вершин n"); axes[1].set_ylabel("интегральный зазор, %")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Время решения и интегральный зазор TSP при росте n (MTZ)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scaling.png"), dpi=150)
    plt.close(fig)

    # --- сравнение LP-релаксаций двух формулировок (доп. задание) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    xpos = np.arange(len(df_sc))
    ax.bar(xpos - 0.2, df_sc["lp_mtz_obj"], 0.4, label="MTZ (big-M)")
    ax.bar(xpos + 0.2, df_sc["lp_flow_obj"], 0.4, label="потоковая (Gavish–Graves)")
    ax.plot(xpos, df_sc["objective"], "k_", ms=14, label="MILP-оптимум")
    ax.set_xticks(xpos, df_sc["scenario"])
    ax.set_ylabel("целевая функция (нижняя граница)")
    ax.set_title("Сила LP-релаксаций: потоковая формулировка точнее MTZ")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lp_strength.png"), dpi=150)
    plt.close(fig)

    print(f"\nрезультаты: {RES_DIR}\nлоги HiGHS: {LOG_DIR}\nграфики: {FIG_DIR}")


if __name__ == "__main__":
    main()
