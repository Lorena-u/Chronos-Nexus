"""Infrastructure selection under a budget (0/1 knapsack family).

Solvers: greedy ratio, exact DP over scaled budget, Branch & Bound with the
fractional (LP) bound, and our modified "core" solver:

  LP-bound variable fixing (reduced-cost fixing) -> shrink the instance ->
  DP or B&B on the remaining core.

Fixing rule (sorted by value/weight, LP(c) = fractional optimum with capacity c):
  * item i in the LP solution is forced IN  if  LP(B + w_i) - v_i < LB
  * item i outside the LP solution is forced OUT if  v_i + LP(B - w_i) < LB
Both bounds are valid upper bounds for solutions violating the fix, so no
optimal solution is lost (LB = value of any feasible solution).
"""
import bisect

from .galaxy import NR, RES_IMP

KINDS = {   # kind: (cost range, build-time range [h], {resource: impact/day}, pop impact, strategic, category)
    "energy_plant": ((300, 900), (48, 120), {2: (40, 120)}, 0, (2, 6), "econ"),
    "farm":         ((200, 700), (72, 168), {0: (30, 90), 1: (-5, -1)}, 0, (1, 4), "econ"),
    "hospital":     ((400, 1100), (96, 200), {4: (10, 40)}, 800, (3, 7), "pop"),
    "factory":      ((500, 1300), (96, 220), {6: (20, 70), 2: (-20, -5)}, 0, (2, 6), "econ"),
    "defense":      ((350, 1000), (72, 160), {}, 0, (6, 12), "defense"),
    "research":     ((450, 1200), (120, 240), {2: (-15, -5)}, 0, (5, 10), "science"),
    "spaceport":    ((600, 1500), (120, 260), {3: (10, 40)}, 0, (4, 9), "expansion"),
    "mining":       ((300, 900), (60, 150), {5: (30, 100)}, 0, (2, 5), "econ"),
}

class Facility:
    __slots__ = ("pid", "kind", "cost", "btime", "res", "pop", "strat", "cat", "value")


def facility_values(g, facs, horizon_h):
    """Value = strategic + need-weighted resource impact + population impact, discounted by build delay."""
    for f in facs:
        p = f.pid
        v = f.strat * (0.5 + g.imp[p])
        for r, imp in f.res.items():
            if imp <= 0:
                continue
            net = g.prod[r][p] - g.cons[r][p]
            need = 1.0 + (3.0 if net < 0 else 0.0)
            v += imp * RES_IMP[r] * 0.05 * need
        v += f.pop * 0.004
        f.value = max(0.0, v * max(0.0, 1.0 - f.btime / horizon_h))

        # ---------------------------------------------------------------------------- solvers
def _greedy(vs, ws, B):
    order = sorted(range(len(vs)), key=lambda i: -vs[i] / ws[i])
    cap, val, ch = B, 0.0, []
    for i in order:
        if ws[i] <= cap:
            cap -= ws[i]; val += vs[i]; ch.append(i)
    best1 = max((i for i in range(len(vs)) if ws[i] <= B), key=lambda i: vs[i], default=None)
    if best1 is not None and vs[best1] > val:
        return vs[best1], [best1]
    return val, ch

def _dp(vs, ws, B, st, cell_limit=40_000_000):
    n = len(vs)
    if n == 0 or B <= 0:
        return 0.0, []
    gran = 1
    while n * (B // gran) > cell_limit:
        gran *= 2
    cap = B // gran
    wi = [max(1, -(-w // gran)) for w in ws]        # round weights UP -> always feasible
    dp = [0.0] * (cap + 1)
    keep = []
    for i in range(n):
        w, v = wi[i], vs[i]
        row = bytearray(cap + 1)
        if w <= cap:
            for c in range(cap, w - 1, -1):
                cand = dp[c - w] + v
                if cand > dp[c]:
                    dp[c] = cand; row[c] = 1
        keep.append(row)
    st.add("dp_cells", n * (cap + 1))
    c, ch = cap, []
    for i in range(n - 1, -1, -1):
        if keep[i][c]:
            ch.append(i); c -= wi[i]
    return dp[cap], ch