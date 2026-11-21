"""Minimum-cost flow (successive shortest paths + Johnson potentials)."""
import heapq

from .galaxy import INF

class MCMF:
    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]
        self.to, self.cap, self.cost = [], [], []

    def add(self, u, v, cap, cost):
        eid = len(self.to)
        self.to.append(v); self.cap.append(cap); self.cost.append(cost); self.g[u].append(eid)
        self.to.append(u); self.cap.append(0); self.cost.append(-cost); self.g[v].append(eid + 1)
        return eid
    def solve(self, s, t, limit=INF):
        """Augment along cheapest paths while marginal path cost < limit."""
        n, g, to, cap, cost = self.n, self.g, self.to, self.cap, self.cost
        pot = [0.0] * n
        flow, total, aug = 0, 0.0, 0
        while True:
            dist = [INF] * n
            dist[s] = 0.0
            pe = [-1] * n
            h = [(0.0, s)]
            while h:
                d, u = heapq.heappop(h)
                if d > dist[u]:
                    continue
                for e in g[u]:
                    if cap[e] > 0:
                        v = to[e]
                        nd = d + cost[e] + pot[u] - pot[v]
                        if nd < dist[v] - 1e-12:
                            dist[v] = nd; pe[v] = e
                            heapq.heappush(h, (nd, v))
            if dist[t] == INF:
                break
            for v in range(n):
                if dist[v] < INF:
                    pot[v] += dist[v]
            if pot[t] - pot[s] >= limit:
                break
            f = INF
            v = t
            while v != s:
                e = pe[v]
                f = min(f, cap[e]); v = to[e ^ 1]
            v = t
            while v != s:
                e = pe[v]
                cap[e] -= f; cap[e ^ 1] += f; v = to[e ^ 1]
            flow += f; total += f * (pot[t] - pot[s]); aug += 1
        return flow, total, aug
    