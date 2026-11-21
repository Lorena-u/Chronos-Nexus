"""Synthetic galaxy / ships / missions / facilities / events generator (seeded)."""
import math
import random

from .galaxy import Galaxy, NR
from .infra import Facility, KINDS

CPC = [12, 10, 15, 6, 2, 3, 4]                    # per-1000-people daily consumption of each resource
DEF_FRAC = [0.02, 0.02, 0.01, 0.005, 0.01, 0.002, 0.002]   # fraction of planets starting in structural deficit

DATASETS = {   # name: (planets, routes, missions)
    "tiny": (30, 120, 15),
    "small": (100, 500, 50),
    "medium": (1000, 10000, 500),
    "large": (10000, 100000, 5000),
    "extreme": (100000, 1000000, 50000),
}

def knn_grid(xs, ys, k, cell):
    n = len(xs)
    grid = {}
    for i in range(n):
        grid.setdefault((int(xs[i] // cell), int(ys[i] // cell)), []).append(i)
    maxr = int(max(max(xs), max(ys)) // cell) + 2
    out = []
    for i in range(n):
        cx, cy = int(xs[i] // cell), int(ys[i] // cell)
        cand = []
        r = 0
        while r <= maxr:
            if r == 0:
                cells = [(cx, cy)]
            else:
                cells = [(cx + dx, cy + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                         if max(abs(dx), abs(dy)) == r]
            for c in cells:
                for j in grid.get(c, ()):
                    if j != i:
                        cand.append(((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2, j))
            if len(cand) >= k:
                cand.sort()
                if (r * cell) ** 2 >= cand[k - 1][0]:
                    break
            r += 1
        cand.sort()
        out.append([j for _, j in cand[:k]])
    return out

class Ship:
    __slots__ = ("id", "loc0", "t0", "loc", "t", "min_t", "cap", "fuel_cap", "speed", "rel", "opc", "caps", "alive")


class Mission:
    __slots__ = ("id", "o", "d", "res", "amt", "release", "deadline", "prio", "imp", "need", "pi",
                 "proactive", "status")   # status: 0 pending 1 scheduled 2 done 3 failed 4 expired

def make_mission(mid, o, d, res, amt, release, deadline, prio, imp, proactive=False):
    from .galaxy import PID
    m = Mission()
    m.id, m.o, m.d, m.res, m.amt = mid, o, d, res, amt
    m.release, m.deadline, m.prio, m.imp = release, deadline, prio, imp
    m.need = 2 if res == 4 else 1
    m.pi = PID["emergency"] if (prio >= 9 or res == 4) else (PID["cargo"] if res in (3, 5, 6) else PID["balanced"])
    m.proactive, m.status = proactive, 0
    return m  

def generate(n_planets, n_routes, n_missions, seed=1, horizon_days=30):
    rng = random.Random(seed)
    g = Galaxy()
    n = n_planets
    side = 1000.0 * math.sqrt(n / 100.0)
    xs = [rng.uniform(0, side) for _ in range(n)]
    ys = [rng.uniform(0, side) for _ in range(n)]
    for i in range(n):
        kind = rng.choices([0, 1, 2], [0.8, 0.12, 0.08])[0]
        pop = int(rng.lognormvariate(math.log(8000), 0.8)) if kind != 2 else int(rng.uniform(50, 500))
        prod, cons, stock, cap = [], [], [], []
        for r in range(NR):
            c = max(1.0, pop / 1000.0 * CPC[r])
            if rng.random() < DEF_FRAC[r]:
                d = c * rng.uniform(0.3, 0.7)
                p = c - d
                s = d * rng.uniform(3, 26)
            else:
                p = c * rng.uniform(1.0, 1.2)
                s = c * rng.uniform(10, 50)
            prod.append(p); cons.append(c); stock.append(s); cap.append(max(500.0, c * 60))
        g.add_planet(xs[i], ys[i], pop, rng.randint(1, 10), rng.randint(0, 10), round(rng.uniform(0.1, 1.0), 2),
                     kind, prod, cons, stock, cap)

     # ---- routes: connectivity chain + k-nearest + wormholes
    kmax = max(8, int(2 * n_routes / n) + 3)
    cell = side / math.sqrt(n / 2.0)
    nb = knn_grid(xs, ys, min(kmax, n - 1), cell)
    pairs = set()
    idx = sorted(range(n), key=lambda i: (int(ys[i] // cell), xs[i] if int(ys[i] // cell) % 2 == 0 else -xs[i]))
    for a, b in zip(idx, idx[1:]):
        pairs.add((min(a, b), max(a, b)))
    n_worm = max(1, int(0.02 * n_routes))
    target = max(len(pairs), n_routes - n_worm)
    done = len(pairs) >= target
    for rank in range(min(kmax, n - 1)):
        if done:
            break
        for i in range(n):
            if rank < len(nb[i]):
                j = nb[i][rank]
                pairs.add((min(i, j), max(i, j)))
                if len(pairs) >= target:
                    done = True
                    break
    for (u, v) in sorted(pairs):
        e = math.hypot(xs[u] - xs[v], ys[u] - ys[v])
        d = e * (1 + rng.uniform(0, 0.25))
        t = d / rng.uniform(8, 12)
        risk = rng.uniform(0.0, 0.012) if rng.random() < 0.85 else rng.uniform(0.04, 0.15)
        g.add_edge(u, v, d, t, t * 0.8, t * 1.5, d * rng.uniform(1.0, 1.8), risk,
                   rng.choice([200, 500, 1000, 2000]), 1 - risk * rng.uniform(0.05, 0.3))
    worm_ids = []
    for _ in range(n_worm):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        e = math.hypot(xs[u] - xs[v], ys[u] - ys[v])
        d = e * 0.25
        t = d / 10.0
        worm_ids.append(g.add_edge(u, v, d, t, t * 0.7, t * 1.8, d * 0.3, rng.uniform(0.05, 0.12), 3000,
                                   rng.uniform(0.9, 0.97), True))

     # ---- spacecraft
    ships = []
    for i in range(max(5, int(n_missions * 0.35))):
        s = Ship()
        s.id = i
        s.loc0 = s.loc = rng.randrange(n)
        s.t0 = s.t = 0.0
        s.min_t = 0.0
        s.cap = rng.uniform(300, 900)
        s.fuel_cap = rng.uniform(900, 2500)
        s.speed = rng.uniform(0.8, 1.5)
        s.rel = rng.uniform(0.93, 0.995)
        s.opc = rng.uniform(1, 4)
        s.caps = 1 | (2 if rng.random() < 0.3 else 0)
        s.alive = True
        ships.append(s)

    
    # ---- dataset missions
    H = horizon_days * 24.0
    missions = []
    for i in range(n_missions):
        o = rng.randrange(n)
        d = o
        for _ in range(3):
            d = rng.choice(nb[d][: max(2, len(nb[d]))])
        if d == o:
            d = nb[o][0]
        res = rng.choices(range(NR), [3, 3, 2, 2, 2, 1, 1])[0]
        rel = rng.uniform(0, H * 0.6) if rng.random() < 0.5 else 0.0
        missions.append(make_mission(i, o, d, res, rng.randint(50, 600), rel, rel + rng.uniform(48, 240),
                                     rng.randint(1, 10), round(rng.uniform(0.5, 2.0), 2)))

      # ---- infrastructure candidates (budget = 20% of total cost)
    facs = []
    for p in rng.sample(range(n), max(1, int(n * 0.3))):
        for kind in rng.sample(list(KINDS), 2):
            (c0, c1), (b0, b1), res, pop, (s0, s1), cat = KINDS[kind]
            f = Facility()
            f.pid, f.kind, f.cost, f.btime = p, kind, rng.uniform(c0, c1), rng.uniform(b0, b1)
            f.res = {r: rng.uniform(*rg) for r, rg in res.items()}
            f.pop, f.strat, f.cat = pop, rng.uniform(s0, s1), cat
            f.value = 0.0
            facs.append(f)
    budget = 0.2 * sum(f.cost for f in facs)

       # ---- dynamic events
    n_ev = max(10, int(10 + (n ** 0.5) * 2.5))
    kinds = ["wormhole_collapse", "route_blocked", "pirate_blockade", "ship_destroyed", "emergency_medical",
             "production_drop", "new_colony", "population_surge", "resource_unavailable", "priority_mission",
             "route_danger", "ship_failure"]
    weights = [1, 3, 3, 2, 2, 3, 1, 2, 2, 2, 3, 2]
    events = []
    for _ in range(n_ev):
        k = rng.choices(kinds, weights)[0]
        t = rng.uniform(24, H - 24)
        sev = round(rng.uniform(0.3, 1.0), 2)
        if k == "wormhole_collapse":
            if not worm_ids:
                continue
            tgt = rng.choice(worm_ids)
        elif k in ("route_blocked", "pirate_blockade", "route_danger"):
            tgt = rng.randrange(g.m)
        elif k in ("ship_destroyed", "ship_failure"):
            tgt = rng.randrange(len(ships))
        elif k == "new_colony":
            tgt = (rng.uniform(0, side), rng.uniform(0, side))
        else:
            tgt = rng.randrange(n)
        events.append((t, k, tgt, sev, rng.randrange(NR)))
    events.sort(key=lambda e: e[0])
    return g, ships, missions, facs, budget, events, nb
