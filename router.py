"""Route planning: Dijkstra, A* with ALT landmarks, Pareto multi-objective search,
route cache with an edge->routes dependency index (used for incremental re-planning).

ALT (A*, Landmarks, Triangle inequality) tables are computed on the *static base
costs of all edges, including currently unavailable ones*.  Hence the lower
bound stays admissible when edges are removed / become more expensive (the usual
dynamic events) and is repaired by an incremental insertion update when new
edges (new colonies) appear.
"""
import heapq
import math
import random

from .galaxy import INF, PID, PNAMES

heappush, heappop = heapq.heappush, heapq.heappop


class Route:
    __slots__ = ("src", "dst", "pi", "arcs", "cost", "time", "tvar", "fuel", "dist", "psucc", "cap")


class Router:
    def __init__(self, g, cfg, st):
        self.g, self.cfg, self.st = g, cfg, st
        self.cache = {}            # (src,dst,profile) -> Route | None
        self.edge_users = {}       # edge -> set(cache keys using it)
        self.lm = {}               # profile -> list of landmark distance tables


# ------------------------------------------------------------------ landmarks
    def _sssp(self, src, costs):
        g = self.g
        dist = [INF] * g.n
        dist[src] = 0.0
        h = [(0.0, src)]
        adj, arc_to = g.adj, g.arc_to
        while h:
            d, u = heappop(h)
            if d > dist[u]:
                continue
            for a in adj[u]:
                v = arc_to[a]
                nd = d + costs[a >> 1]
                if nd < dist[v]:
                    dist[v] = nd
                    heappush(h, (nd, v))
        return dist

    def _ensure_lm(self, pi):
        if pi in self.lm:
            return self.lm[pi]
        g, L = self.g, self.cfg.landmarks
        costs = g.base[pi]
        tabs = []
        rng = random.Random(7)
        first = self._sssp(rng.randrange(g.n), costs)
        cur = max(range(g.n), key=lambda v: first[v] if first[v] < INF else -1)
        mind = [INF] * g.n
        for _ in range(L):                       # farthest-first landmark selection
            tab = self._sssp(cur, costs)
            tabs.append(tab)
            for v in range(g.n):
                if tab[v] < mind[v]:
                    mind[v] = tab[v]
            cur = max(range(g.n), key=lambda v: mind[v] if mind[v] < INF else -1)
        self.lm[pi] = tabs
        return tabs

    def landmarks_insert(self, new_edges):
        """Dynamic-graph update: relax landmark tables after edge/node insertions."""
        g = self.g
        for pi, tabs in self.lm.items():
            costs = g.base[pi]
            for tab in tabs:
                if len(tab) < g.n:
                    tab.extend([INF] * (g.n - len(tab)))
                heap = []
                for e in new_edges:
                    u, v, c = g.eu[e], g.ev[e], costs[e]
                    if tab[u] + c < tab[v]:
                        tab[v] = tab[u] + c; heappush(heap, (tab[v], v))
                    if tab[v] + c < tab[u]:
                        tab[u] = tab[v] + c; heappush(heap, (tab[u], u))
                while heap:
                    d, u = heappop(heap)
                    if d > tab[u]:
                        continue
                    for a in g.adj[u]:
                        v = g.arc_to[a]
                        nd = d + costs[a >> 1]
                        if nd < tab[v]:
                            tab[v] = nd; heappush(heap, (nd, v))

# ------------------------------------------------------------------ point to point
    def _use_alt(self):
        a = self.cfg.route_algo
        return a == "astar" or (a == "auto" and self.g.n >= 300)

    def _astar(self, src, dst, pi, use_h):
        g = self.g
        adj, arc_to, avail, cost = g.adj, g.arc_to, g.avail, g.cost[pi]
        tabs = []
        if use_h:
            for tab in self._ensure_lm(pi):
                if tab[dst] < INF:
                    tabs.append((tab, tab[dst]))

        def h(v):
            b = 0.0
            for tab, dt in tabs:
                a = tab[v]
                if a < INF:
                    d = dt - a
                    if d < 0:
                        d = -d
                    if d > b:
                        b = d
            return b

        best = {src: 0.0}
        prev = {}
        closed = set()
        heap = [(h(src), 0.0, src)]
        explored = 0
        found = False
        while heap:
            _, gv, u = heappop(heap)
            if u in closed:
                continue
            closed.add(u)
            explored += 1
            if u == dst:
                found = True
                break
            for a in adj[u]:
                e = a >> 1
                if not avail[e]:
                    continue
                v = arc_to[a]
                if v in closed:
                    continue
                ng = gv + cost[e]
                if ng < best.get(v, INF):
                    best[v] = ng
                    prev[v] = a
                    heappush(heap, (ng + h(v), ng, v))
        if not found:
            return None, explored
        arcs = []
        v = dst
        while v != src:
            a = prev[v]
            arcs.append(a)
            v = arc_to[a ^ 1]
        arcs.reverse()
        return self._mk(src, dst, pi, arcs, best[dst]), explored
                
    def _mk(self, src, dst, pi, arcs, cost):
        g = self.g
        r = Route()
        r.src, r.dst, r.pi, r.arcs, r.cost = src, dst, pi, arcs, cost
        t = f = d = 0.0
        var = 0.0
        ps = 1.0
        cap = float("inf")
        for a in arcs:
            e = a >> 1
            cap = min(cap, g.ecap[e])
            t += g.time[e]; f += g.fuel[e]; d += g.dist[e]
            var += ((g.tmax[e] - g.tmin[e]) * 0.25) ** 2
            ps *= g.psucc(e)
        r.time, r.fuel, r.dist, r.tvar, r.psucc, r.cap = t, f, d, var, ps, cap
        return r
    