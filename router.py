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

    