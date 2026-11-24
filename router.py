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

    def _register(self, key, route):
        if route is None:
            return
        for a in route.arcs:
            self.edge_users.setdefault(a >> 1, set()).add(key)
    
    def search(self, src, dst, pi, force=None):
        key = (src, dst, pi)
        if key in self.cache:
            self.st.add("route_cache_hits")
            return self.cache[key]
        use_h = self._use_alt() if force is None else force
        if src == dst:
            r = self._mk(src, dst, pi, [], 0.0)
        else:
            r, ex = self._astar(src, dst, pi, use_h)
            self.st.add("nodes_explored", ex)
            self.st.add("route_queries")
        self.cache[key] = r
        self._register(key, r)
        return r

    # ------------------------------------------------------------------ Pareto (multi-objective)
    def pareto(self, src, dst, eps=0.04, cap=6, max_labels=20000):
        """Label-setting search over (time, fuel, -log P(success)).

        Search-space reduction: epsilon-dominance (labels at a node must differ by
        >4% in some objective), at most `cap` labels per node, ellipse corridor
        around src-dst, and pruning by labels already settled at the destination.
        """
        g = self.g
        d0 = g.euclid(src, dst)
        lim = 1.5 * d0 + 60.0
        adj, arc_to, avail = g.adj, g.arc_to, g.avail
        labels = {src: [(0.0, 0.0, 0.0)]}
        heap = [(0.0, 0, (0.0, 0.0, 0.0), src, None, -1)]
        nid = 1
        goal = []
        out = []
        pops = 0
        while heap and pops < max_labels:
            s, _, (t, f, r), u, parent, arc = heappop(heap)
            pops += 1
            node = (t, f, r, u, parent, arc)
            if u == dst:
                if not any(o[0] <= t and o[1] <= f and o[2] <= r for o in goal):
                    goal.append((t, f, r)); out.append(node)
                continue
            for a in adj[u]:
                e = a >> 1
                if not avail[e]:
                    continue
                v = arc_to[a]
                if g.euclid(src, v) + g.euclid(v, dst) > lim:
                    continue
                nt = t + g.time[e] + (g.tmax[e] - g.tmin[e]) * 0.25
                nf = f + g.fuel[e]
                nr = r - math.log(max(g.psucc(e), 1e-9))
                if any(o[0] <= nt and o[1] <= nf and o[2] <= nr for o in goal):
                    continue
                lv = labels.setdefault(v, [])
                dom = False
                for (lt, lf, lr) in lv:
                    if lt <= nt * (1 + eps) and lf <= nf * (1 + eps) and lr <= nr * (1 + eps) + 1e-9:
                        dom = True
                        break
                if dom or len(lv) >= cap:
                    continue
                lv.append((nt, nf, nr))
                heappush(heap, (nt / 50.0 + nf / 100.0 + nr * 10.0, nid, (nt, nf, nr), v, node, a))
                nid += 1
        self.st.add("pareto_labels", pops)
        # final filter: drop labels dominated by a later-settled one
        out = [o for o in out if not any(q is not o and q[0] <= o[0] and q[1] <= o[1] and q[2] <= o[2] and
                                          (q[0], q[1], q[2]) != (o[0], o[1], o[2]) for q in out)]        
        routes = []
        for (t, f, r, u, parent, arc) in out:
            arcs = []
            node = (t, f, r, u, parent, arc)
            while node[4] is not None:
                arcs.append(node[5]); node = node[4]
            arcs.reverse()
            routes.append(self._mk(src, dst, PID["emergency"], arcs, t))
        return routes

    def emergency_route(self, src, dst):
        key = (src, dst, 3)           # 3 = "pareto" pseudo-profile
        if key in self.cache:
            return self.cache[key]
        routes = self.pareto(src, dst)
        if not routes:
            r = self.search(src, dst, PID["emergency"])
        else:
            best_p = max(x.psucc for x in routes)
            ok = [x for x in routes if x.psucc >= best_p * 0.95]
            r = min(ok, key=lambda x: x.time + 2.0 * math.sqrt(x.tvar))
        self.cache[key] = r
        self._register(key, r)
        return r

    # ------------------------------------------------------------------ flow candidate search
    def nearest_targets(self, src, pi, pred, k, maxcost):
        """Dijkstra stopped after k target nodes (satisfying pred) are settled."""
        g = self.g
        adj, arc_to, avail, cost = g.adj, g.arc_to, g.avail, g.cost[pi]
        dist = {src: 0.0}
        done = set()
        heap = [(0.0, src)]
        res = []
        ex = 0
        while heap and len(res) < k:
            d, u = heappop(heap)
            if u in done:
                continue
            done.add(u); ex += 1
            if d > maxcost:
                break
            if u != src and pred(u):
                res.append((u, d))
            for a in adj[u]:
                e = a >> 1
                if not avail[e]:
                    continue
                v = arc_to[a]
                nd = d + cost[e]
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    heappush(heap, (nd, v))
        self.st.add("nodes_explored", ex)
        return res

    # ------------------------------------------------------------------ dynamic updates
    def invalidate_edges(self, edges):
        """Drop every cached route that traverses one of `edges`; return their keys."""
        keys = set()
        for e in edges:
            ks = self.edge_users.pop(e, None)
            if ks:
                keys |= ks
        for k in keys:
            r = self.cache.pop(k, None)
            if r is not None:
                for a in r.arcs:
                    s = self.edge_users.get(a >> 1)
                    if s:
                        s.discard(k)
        return keys

    def clear(self):
        self.cache.clear(); self.edge_users.clear()

    def drop_failed(self):
        self.cache = {k: v for k, v in self.cache.items() if v is not None}
    