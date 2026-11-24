"""Prediction + resource allocation.

* first_shortage / need: piecewise-linear stock forecast that accounts for
  deliveries already planned (so we never double-order).
* A min-heap of predicted shortage times is kept per (resource, planet) and
  repaired lazily when production/consumption/deliveries change.
* allocate(): min-cost flow  supply planets -> shortage planets  with a per-unit
  shortage penalty; candidate arcs are pruned to the K cheapest suppliers.
"""
import heapq
import math

from .flow import MCMF
from .galaxy import INF, NR, PID, RES_IMP, SURV_W
from .generator import make_mission

CHUNK = 300.0          # cargo per mission
UNIT_REF = 300.0       # a route cost is spread over one shipload
PENALTY = 20.0         # shortage penalty per unit (scaled by importance)
RESERVE_DAYS = 10      # supplier keeps this much stock

class ResourcePlanner:
    def __init__(self, g, router, sched, cfg, st):
        self.g, self.rt, self.sched, self.cfg, self.st = g, router, sched, cfg, st
        self.heap = []

    # ------------------------------------------------------------ forecasting
    def first_shortage(self, r, p, now):
        """Hours from now until stock hits zero (None if never within projection)."""
        g = self.g
        net = (g.prod[r][p] - g.cons[r][p]) / 24.0
        s, tc = g.stock[r][p], 0.0
        for ta, amt in self.sched.inbound(r, p, now):
            if net < 0 and s + net * (ta - tc) < 0:
                return tc + s / (-net)
            s += net * (ta - tc) + amt
            tc = ta
        if net < 0:
            return tc + max(0.0, s) / (-net)
        return None
    def need(self, r, p, now, window_h):
        """Units that must still arrive so the stock stays >= 0 during `window_h`."""
        g = self.g
        net = (g.prod[r][p] - g.cons[r][p]) / 24.0
        s, tc, low = g.stock[r][p], 0.0, g.stock[r][p]
        for ta, amt in self.sched.inbound(r, p, now):
            if ta > window_h:
                break
            s += net * (ta - tc)
            low = min(low, s)
            s += amt
            tc = ta
        s += net * (window_h - tc)
        low = min(low, s)
        return max(0.0, -low)

    def cover(self, now, lookahead_h):
        """Hysteresis: trigger at `lookahead`, but order enough to cover twice that window."""
        return max(24.0, min(2.0 * lookahead_h, self.cfg.horizon_days * 24.0 - now))

    def build_heap(self, now):
        g = self.g
        self.heap = []
        for r in range(NR):
            pr, co, sk = g.prod[r], g.cons[r], g.stock[r]
            for p in range(g.n):
                if pr[p] < co[p]:
                    heapq.heappush(self.heap, (now + sk[p] / ((co[p] - pr[p]) / 24.0), r, p))
        self.st.add("forecast_entries", len(self.heap))

    def touch(self, r, p, now):
        g = self.g
        if g.prod[r][p] < g.cons[r][p]:
            heapq.heappush(self.heap, (now + g.stock[r][p] / ((g.cons[r][p] - g.prod[r][p]) / 24.0), r, p))

    def scan(self, now, lookahead_h):
        """Pop every entry predicted to run dry within the look-ahead; return uncovered demands."""
        out, seen = [], set()
        horizon = now + lookahead_h
        keep = []
        while self.heap and self.heap[0][0] <= horizon:
            t, r, p = heapq.heappop(self.heap)
            if (r, p) in seen:
                continue
            seen.add((r, p))
            ts = self.first_shortage(r, p, now)
            if ts is None or ts > lookahead_h:
                if ts is not None:
                    keep.append((now + ts, r, p))
                continue
            need = self.need(r, p, now, self.cover(now, lookahead_h))
            if need >= 20:
                out.append((r, p, ts, need))
            keep.append((now + ts, r, p))
        for e in keep:
            heapq.heappush(self.heap, e)
        return out

    # ------------------------------------------------------------ allocation
    def _avail(self, r, v):
        g = self.g
        if g.prod[r][v] < g.cons[r][v]:
            return 0.0
        return g.stock[r][v] - RESERVE_DAYS * g.cons[r][v] - self.sched.outbound(r, v)

    def allocate(self, demands, now):
        """demands: [(r, planet, t_short_h, need)] -> list of new Mission objects."""
        g, st = self.g, self.st
        by_r = {}
        for d in demands:
            by_r.setdefault(d[0], []).append(d)
        shipments = []
        for r, ds in by_r.items():
            arcs, sup_ids, dem_ids = [], {}, {}
            naive_s = 0
            for (_, p, ts, need) in ds:
                cands = self.rt.nearest_targets(p, PID["cargo"], lambda v, r=r: self._avail(r, v) > 20,
                                                self.cfg.flow_k, 4000.0)

                for (s, c) in cands:
                    arcs.append((s, p, c))
            sup = sorted({a[0] for a in arcs})
            sup_ids = {s: 2 + i for i, s in enumerate(sup)}
            dem_ids = {d[1]: 2 + len(sup) + i for i, d in enumerate(ds)}
            st.add("flow_arcs", len(arcs))
            st.add("naive_flow_arcs", g.n * len(ds))      # full supply x demand bipartite graph
            if self.cfg.flow_mode == "greedy":
                shipments += self._greedy_alloc(r, ds, arcs, now)
                continue
            net = MCMF(2 + len(sup) + len(ds))
            for s in sup:
                net.add(0, sup_ids[s], int(self._avail(r, s)), 0.0)
            pmax = PENALTY * RES_IMP[r] * 2.0
            for (_, p, ts, need) in ds:
                pen = PENALTY * RES_IMP[r] * (0.5 + g.imp[p])
                net.add(dem_ids[p], 1, int(math.ceil(need)), pmax - pen)
            arc_ids = []
            for (s, p, c) in arcs:
                arc_ids.append((net.add(sup_ids[s], dem_ids[p], 10 ** 9, c / UNIT_REF), s, p))
            flow, cost, aug = net.solve(0, 1, limit=pmax)
            st.add("flow_augmentations", aug)
            info = {d[1]: d for d in ds}
            for eid, s, p in arc_ids:
                f = net.cap[eid ^ 1]
                if f >= 10:
                    shipments.append((r, s, p, f, info[p][2]))

             # turn shipments into chunked missions
        out = []
        for (r, s, p, amt, ts) in shipments:
            left = amt
            while left >= 10:
                a = min(CHUNK, left)
                left -= a
                dl = now + max(12.0, ts - 12.0)
                prio = int(min(10, 4 + 5 * SURV_W[r] + (1 if ts < 48 else 0) + g.imp[p]))
                m = make_mission(len(self.sched.missions) + len(out), s, p, r, a, now, dl, prio,
                                 round(1.0 + g.imp[p], 2), proactive=True)
                out.append(m)
        return out
    def _greedy_alloc(self, r, ds, arcs, now):
        """Baseline: serve demands by decreasing importance from the nearest supplier."""
        left = {}
        res = []
        for (_, p, ts, need) in sorted(ds, key=lambda d: -self.g.imp[d[1]]):
            rem = need
            for (s, q, c) in sorted((a for a in arcs if a[1] == p), key=lambda a: a[2]):
                have = left.setdefault(s, self._avail(r, s))
                take = min(have, rem)
                if take >= 10:
                    left[s] -= take; rem -= take
                    res.append((r, s, p, take, ts))
                if rem < 10:
                    break
        return res



                