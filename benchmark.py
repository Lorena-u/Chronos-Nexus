"""Experiments for the report.  python benchmark.py [--quick] [--large] [--xl]

Writes results/*.json, results/*.md and results/*.png
"""
import argparse
import json
import os
import random
import statistics
import time

from chronos.config import Config
from chronos.galaxy import PID
from chronos.generator import DATASETS, generate
from chronos.infra import Facility, facility_values, lp_upper_bound, solve
from chronos.planner import ResourcePlanner
from chronos.sim import Simulation
from chronos.stats import Stats
from main import baseline_cfg

OUT = "results"
os.makedirs(OUT, exist_ok=True)
R = {}

def save(name, obj):
    R[name] = obj
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(obj, f, indent=1)

def table(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return s

def run(ds, seed=1, sizes=None, **kw):
    cfg = Config(seed=seed)
    base = kw.pop("baseline", False)
    if base:
        baseline_cfg(cfg)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return Simulation(cfg, ds, sizes=sizes).run()

# ------------------------------------------------------------------ E1 scalability
def e1_scalability(datasets, sizes_extra):
    rows = []
    for ds in datasets:
        r = run(ds)
        rows.append(r)
    for name, sz in sizes_extra.items():
        r = run(name, sizes=sz)
        rows.append(r)
    save("e1_scalability", [{k: r[k] for k in (
        "dataset", "planets", "routes", "missions_total", "spacecraft", "civilization_score", "missions_completed",
        "missions_failed", "population_survival_rate", "shortage_units", "states_explored", "graph_nodes_explored",
        "search_space_reduction_pct", "avg_replan_time_ms", "total_execution_time_s", "generation_time_s",
        "peak_memory_mb", "events_handled", "scheduler_used")} for r in rows])


# ------------------------------------------------------------------ E2 baseline vs optimised
def e2_baseline(datasets, seeds):
    rows = []
    for ds in datasets:
        for base in (True, False):
            rs = [run(ds, seed=s, baseline=base) for s in seeds]
            rows.append({
                "dataset": ds, "config": "naive baseline" if base else "Chronos (ours)",
                "score": round(statistics.mean(r["civilization_score"] for r in rs), 1),
                "time_s": round(statistics.mean(r["total_execution_time_s"] for r in rs), 2),
                "mem_MB": round(statistics.mean(r["peak_memory_mb"] for r in rs), 1),
                "states": int(statistics.mean(r["states_explored"] for r in rs)),
                "replan_ms": round(statistics.mean(r["avg_replan_time_ms"] for r in rs), 2),
                "shortage": round(statistics.mean(r["shortage_units"] for r in rs)),
                "completed": round(statistics.mean(r["missions_completed"] for r in rs))})
    save("e2_baseline", rows)

# ------------------------------------------------------------------ E3 routing
def e3_routing(datasets, nq=300):
    rows = []
    for ds in datasets:
        sim = Simulation(Config(seed=1), ds)
        rt, g = sim.rt, sim.g
        t = time.perf_counter(); rt._ensure_lm(PID["balanced"]); lm_t = time.perf_counter() - t
        rng = random.Random(9)
        qs = [(rng.randrange(g.n), rng.randrange(g.n)) for _ in range(nq)]
        res = {}
        for name, use in (("Dijkstra", False), ("A* + ALT", True)):
            t = time.perf_counter(); nodes = 0; mx = 0; costs = []
            for s, d in qs:
                r, ex = rt._astar(s, d, PID["balanced"], use)
                nodes += ex; mx = max(mx, ex); costs.append(r.cost if r else None)
            res[name] = (time.perf_counter() - t, nodes, mx, costs)
        assert all((a is None and b is None) or abs(a - b) < 1e-6 for a, b in zip(res["Dijkstra"][3], res["A* + ALT"][3]))
        # Pareto
        t = time.perf_counter(); labels0 = sim.st.c["pareto_labels"]; sizes = []
        for s, d in qs[:40]:
            if g.euclid(s, d) < 600:
                sizes.append(len(rt.pareto(s, d)))
        pt = time.perf_counter() - t
        rows.append({"dataset": ds, "nodes": g.n, "edges": g.m,
                     "dijkstra_nodes_per_query": round(res["Dijkstra"][1] / nq), "alt_nodes_per_query": round(res["A* + ALT"][1] / nq),
                     "reduction_pct": round(100 * (1 - res["A* + ALT"][1] / res["Dijkstra"][1]), 1),
                     "dijkstra_ms_per_query": round(1000 * res["Dijkstra"][0] / nq, 3),
                     "alt_ms_per_query": round(1000 * res["A* + ALT"][0] / nq, 3),
                     "worst_case_nodes_dijkstra": res["Dijkstra"][2], "worst_case_nodes_alt": res["A* + ALT"][2],
                     "landmark_build_s": round(lm_t, 3), "pareto_avg_routes": round(statistics.mean(sizes), 2) if sizes else 0,
                     "pareto_ms_per_query": round(1000 * pt / max(1, len(sizes)), 2), "costs_identical": True})
    save("e3_routing", rows)

    