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