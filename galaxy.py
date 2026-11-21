"""Dynamic weighted graph G=(V,E) plus planet / resource model.

Edges are undirected routes stored as two arcs (2e: u->v, 2e+1: v->u).
Per-profile edge costs are cached in flat lists so that an event touching one
edge costs O(#profiles) to apply.
"""
import math

RES = ["food", "water", "energy", "fuel", "medicine", "minerals", "materials"]
NR = len(RES)
RES_IMP = [1.0, 1.0, 0.6, 0.5, 1.2, 0.3, 0.3]       # importance of each resource
SURV_W = [1.0, 1.0, 0.3, 0.0, 0.8, 0.0, 0.0]        # contribution to population survival
# weights (w_dist, w_fuel, w_time, w_risk) of Cost = w1*Dist + w2*Fuel + w3*Time + w4*FailProb
PROFILES = {"cargo": (1.0, 1.0, 0.3, 40.0),
            "emergency": (0.1, 0.1, 2.0, 150.0),
            "balanced": (0.5, 0.5, 0.5, 80.0)}
PNAMES = list(PROFILES)
PID = {n: i for i, n in enumerate(PNAMES)}
INF = float("inf")

class Galaxy:
    def __init__(self):
        # planets / colonies / stations
        self.x, self.y = [], []
        self.pop, self.tech, self.defense, self.imp, self.kind = [], [], [], [], []
        self.prod = [[] for _ in range(NR)]
        self.cons = [[] for _ in range(NR)]
        self.stock = [[] for _ in range(NR)]
        self.cap = [[] for _ in range(NR)]
        # routes (edge arrays)
        self.eu, self.ev = [], []
        self.dist, self.time, self.tmin, self.tmax = [], [], [], []
        self.fuel, self.risk, self.ecap, self.rel = [], [], [], []
        self.avail, self.worm = [], []
        self.adj, self.arc_to = [], []
        self.cost = [[] for _ in PNAMES]    # current per-profile cost
        self.base = [[] for _ in PNAMES]    # cost at creation (static lower bound for ALT)

    @property
    def n(self):
        return len(self.x)

    @property
    def m(self):
        return len(self.eu)

    def add_planet(self, x, y, pop, tech, defense, imp, kind, prod, cons, stock, cap):
        self.x.append(x); self.y.append(y); self.pop.append(pop); self.tech.append(tech)
        self.defense.append(defense); self.imp.append(imp); self.kind.append(kind)
        for r in range(NR):
            self.prod[r].append(prod[r]); self.cons[r].append(cons[r])
            self.stock[r].append(stock[r]); self.cap[r].append(cap[r])
        self.adj.append([])
        return len(self.x) - 1