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

def _bnb(vs, ws, B, st, node_limit, lb0, ch0):
    """Depth-first Branch & Bound on ratio-sorted items with fractional upper bound."""
    n = len(vs)
    order = sorted(range(n), key=lambda i: -vs[i] / ws[i])
    V = [vs[i] for i in order]; W = [ws[i] for i in order]

    def bound(i, cap, val):
        while i < n:
            if W[i] <= cap:
                cap -= W[i]; val += V[i]
            else:
                return val + V[i] * cap / W[i]
            i += 1
        return val

    best, bestset, nodes = lb0, list(ch0), 0
    stack = [(0, B, 0.0, None)]
    while stack and nodes < node_limit:
        i, cap, val, link = stack.pop()
        nodes += 1
        if val > best:
            best = val
            bestset = []
            l = link
            while l is not None:
                bestset.append(order[l[0]]); l = l[1]
        if i >= n or bound(i, cap, val) <= best + 1e-12:
            continue
        stack.append((i + 1, cap, val, link))                      # exclude i
        if W[i] <= cap:
            stack.append((i + 1, cap - W[i], val + V[i], (i, link)))  # include i (explored first)
    st.add("bnb_nodes", nodes)
    return best, bestset, nodes >= node_limit

def _core(vs, ws, B, st, mode):
    n = len(vs)
    order = sorted(range(n), key=lambda i: -vs[i] / ws[i])
    V = [vs[i] for i in order]; W = [ws[i] for i in order]
    PW, PV = [0], [0.0]
    for i in range(n):
        PW.append(PW[-1] + W[i]); PV.append(PV[-1] + V[i])

    def lp(cap):
        if cap <= 0:
            return 0.0
        k = bisect.bisect_right(PW, cap) - 1
        if k >= n:
            return PV[n]
        return PV[k] + (cap - PW[k]) * V[k] / W[k]

    lb, lbset = _greedy(vs, ws, B)
    c = bisect.bisect_right(PW, B) - 1       # first item that does not fit fully
    fixed_in, fixed_out, core = [], [], []
    for pos in range(n):
        i = order[pos]
        if pos < c:
            if lp(B + W[pos]) - V[pos] < lb - 1e-9:
                fixed_in.append(i); continue
        else:
            ub = V[pos] + lp(B - W[pos]) if B - W[pos] >= 0 else -1.0
            if ub < lb - 1e-9:
                fixed_out.append(i); continue
        core.append(i)
    st.add("knap_fixed", len(fixed_in) + len(fixed_out))
    B2 = B - sum(ws[i] for i in fixed_in)
    cv = [vs[i] for i in core]; cw = [ws[i] for i in core]
    base = sum(vs[i] for i in fixed_in)
    if not core:
        return base, fixed_in, "fixing-only"
    cells = len(core) * max(1, B2 // max(1, B // 4000))
    if mode in ("dp",) or (mode == "core" and cells <= 6_000_000):
        v, ch = _dp(cv, cw, B2, st); how = "core+DP"
    else:
        g0 = _greedy(cv, cw, B2)
        v, ch, trunc = _bnb(cv, cw, B2, st, 150_000, g0[0], g0[1]); how = "core+B&B" + ("(truncated)" if trunc else "")
    return base + v, fixed_in + [core[j] for j in ch], how

def solve(facs, budget, mode, st, scale=10):
    """Return (chosen facility indices, total value, method)."""
    items = [i for i, f in enumerate(facs) if f.value > 0 and f.cost <= budget]
    vs = [facs[i].value for i in items]
    ws = [max(1, -(-int(facs[i].cost * 100) // (scale * 100))) for i in items]   # ceil: never exceed budget
    B = int(budget // scale)
    n = len(items)
    st.add("knap_items", n)
    st.add("naive_knap_cells", n * B)           # the full DP table a naive exact solver would fill
    if n == 0:
        return [], 0.0, "empty"
    if mode == "auto":
        mode = "core"
    if mode == "greedy":
        v, ch = _greedy(vs, ws, B); how = "greedy"
    elif mode == "dp":
        v, ch = _dp(vs, ws, B, st); how = "dp"
    elif mode == "bnb":
        g0 = _greedy(vs, ws, B)
        v, ch, tr = _bnb(vs, ws, B, st, 400_000, g0[0], g0[1]); how = "bnb" + ("(truncated)" if tr else "")
    else:
        v, ch, how = _core(vs, ws, B, st, "core")
    chosen = [items[i] for i in ch]
    return chosen, sum(facs[i].value for i in chosen), how

def lp_upper_bound(facs, budget):
    items = sorted((f for f in facs if f.value > 0), key=lambda f: -f.value / f.cost)
    cap, val = budget, 0.0
    for f in items:
        if f.cost <= cap:
            cap -= f.cost; val += f.value
        else:
            val += f.value * cap / f.cost
            break
    return val
