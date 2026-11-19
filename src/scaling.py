"""Кривая «время решения / зазор» при росте числа вершин (10, 12, 20, 40).

Данные для каждого n генерируются тем же seed=10204 (размер n входит
в поток генератора, поэтому наборы точек различны и воспроизводимы).
"""

import numpy as np

from .generate_data import generate
from .model import solve_mtz, solve_flow

SIZES = [10, 12, 20, 40]


def scaling_study(time_limit=120.0, sizes=SIZES):
    rows = []
    for n in sizes:
        _, dist = generate(n_points=n)
        res_milp, t_milp = solve_mtz(dist, relax=False, time_limit=time_limit)
        res_lp, t_lp = solve_mtz(dist, relax=True, time_limit=time_limit)
        res_flow, t_flow = solve_flow(dist, relax=True, time_limit=time_limit)
        obj = res_milp.fun if res_milp.success else np.nan
        lp_mtz = res_lp.fun
        lp_flow = res_flow.fun
        gap = (obj - max(lp_mtz, lp_flow)) / obj if np.isfinite(obj) and obj else np.nan
        rows.append({
            "n": n,
            "milp_obj": obj,
            "milp_time_s": round(t_milp, 4),
            "milp_nodes": getattr(res_milp, "mip_node_count", np.nan),
            "lp_mtz_obj": lp_mtz,
            "lp_mtz_time_s": round(t_lp, 4),
            "lp_flow_obj": lp_flow,
            "lp_flow_time_s": round(t_flow, 4),
            "integrality_gap": gap,
        })
    return rows
