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