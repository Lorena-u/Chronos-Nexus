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

# ------------------------------------------------------------------ E4 scheduling (hybrid ablation)
def e4_scheduling(ds, budget, seeds):
    rows, hist = [], {}
    for mode in ("greedy", "greedy_ls", "ga", "ga_ls"):
        objs, ts, dec, civ = [], [], [], []
        for s in seeds:
            cfg = Config(seed=s, sched_mode=mode, time_budget_s=budget)
            sim = Simulation(cfg, ds)
            t = time.perf_counter(); sim.initial_plan(); dt = time.perf_counter() - t
            objs.append(sim.sched.last_obj); ts.append(sim.sched_time); dec.append(sim.st.c["decodes"])
            if s == seeds[0]:
                hist[mode] = {"ga": sim.sched.last_hist, "ls": sim.sched.last_ls_hist}
            civ.append(sim.sched.greedy_value)
        rows.append({"method": mode, "objective(mean)": round(statistics.mean(objs), 1),
                     "gain_vs_greedy_%": round(100 * (statistics.mean(objs) / statistics.mean(civ) - 1), 2),
                     "sched_time_s": round(statistics.mean(ts), 2), "full_schedules_evaluated": int(statistics.mean(dec))})
    save("e4_scheduling", {"dataset": ds, "budget_s": budget, "rows": rows, "history": hist})   

# ------------------------------------------------------------------ E5 knapsack
def e5_infra():
    rows = []
    rng = random.Random(4)

    def inst(n, kind):
        fs = []
        for _ in range(n):
            f = Facility(); f.cost = rng.uniform(100, 1000)
            f.value = f.cost * 0.05 * rng.uniform(0.7, 1.3) if kind == "random-ratio" else (f.cost * 0.05 + 5 if kind == "strongly-correlated" else rng.uniform(1, 60))
            fs.append(f)
        return fs

    for n, kind in ((200, "random"), (1000, "random"), (3000, "random"), (3000, "random-ratio"), (1500, "strongly-correlated")):
        fs = inst(n, kind)
        budget = 0.25 * sum(f.cost for f in fs)
        lp = lp_upper_bound(fs, budget)
        for mode in ("greedy", "bnb", "dp", "core"):
            if mode == "dp" and n > 1500:
                pass
            st = Stats(); t = time.perf_counter()
            ch, val, how = solve(fs, budget, mode, st)
            dt = time.perf_counter() - t
            rows.append({"n": n, "instance": kind, "solver": mode, "value": round(val, 2), "gap_to_LP_%": round(100 * (lp - val) / lp, 3),
                         "time_s": round(dt, 3), "dp_cells": int(st.c["dp_cells"]), "bnb_nodes": int(st.c["bnb_nodes"]),
                         "items_fixed_%": round(100 * st.c["knap_fixed"] / n, 1), "method": how})
    save("e5_infra", rows)

# ------------------------------------------------------------------ E6 flow vs greedy allocation (contention)
def e6_flow(ds):
    rows = []
    for scarcity in (1.0, 0.08):          # 1.0 = ample supply, 0.08 = supplies scarce (contention)
        for mode in ("greedy", "mcmf"):
            cfg = Config(seed=1, flow_mode=mode, lookahead_days=30)
            sim = Simulation(cfg, ds)
            g = sim.g
            for r in range(7):
                for p in range(g.n):
                    if g.prod[r][p] >= g.cons[r][p]:
                        g.stock[r][p] = 10 * g.cons[r][p] + (g.stock[r][p] - 10 * g.cons[r][p]) * scarcity
            sim.planner.build_heap(0.0)
            demands = sim.planner.scan(0.0, 30 * 24.0)
            t = time.perf_counter()
            ms = sim.planner.allocate(demands, 0.0)
            dt = time.perf_counter() - t
            shipped = sum(m.amt for m in ms)
            cost = penalty = 0.0
            from chronos.planner import PENALTY
            from chronos.galaxy import RES_IMP
            got = {}
            for m in ms:
                r = sim.rt.search(m.o, m.d, PID["cargo"])
                cost += (r.cost / 300.0) * m.amt if r else 0
                got[(m.res, m.d)] = got.get((m.res, m.d), 0) + m.amt
            for (r_, p, ts, need) in demands:
                penalty += max(0.0, need - got.get((r_, p), 0)) * PENALTY * RES_IMP[r_] * (0.5 + g.imp[p])
            need = sum(d[3] for d in demands)
            rows.append({"supply": "ample" if scarcity == 1.0 else "scarce", "method": "min-cost flow" if mode == "mcmf" else "greedy nearest",
                         "demands": len(demands), "units_needed": round(need), "unmet_units": round(max(0, need - shipped)),
                         "transport_cost": round(cost), "shortage_penalty": round(penalty), "total_objective": round(cost + penalty),
                         "time_s": round(dt, 3)})
    save("e6_flow", rows)

# ------------------------------------------------------------------ E7 replanning + proactive
def e7_dynamic(ds, seeds):
    rows = []
    for rp in ("incremental", "full"):
        rs = [run(ds, seed=s, replan=rp) for s in seeds]
        rows.append({"replanning": rp, "avg_replan_ms": round(statistics.mean(r["avg_replan_time_ms"] for r in rs), 2),
                     "max_replan_ms": round(max(r["max_replan_time_ms"] for r in rs), 1),
                     "missions_recomputed": round(statistics.mean(r["missions_recomputed_by_events"] for r in rs)),
                     "score": round(statistics.mean(r["civilization_score"] for r in rs), 1),
                     "total_time_s": round(statistics.mean(r["total_execution_time_s"] for r in rs), 2)})
    save("e7_replan", rows)
    rows = []
    for la, name in ((14, "proactive (14-day look-ahead)"), (3, "short look-ahead (3 days)"), (1, "reactive (1 day)")):
        rs = [run(ds, seed=s, lookahead_days=la) for s in seeds]
        rows.append({"policy": name, "shortage_units": round(statistics.mean(r["shortage_units"] for r in rs)),
                     "shortage_pairs": round(statistics.mean(r["shortage_planet_resource_pairs"] for r in rs)),
                     "survival": round(statistics.mean(r["population_survival_rate"] for r in rs), 5),
                     "score": round(statistics.mean(r["civilization_score"] for r in rs), 1)})
    save("e7_proactive", rows)

def plots():
    for f in os.listdir(OUT):
        if f.endswith(".json") and f[:-5] not in R and f[:2] in ("e1","e2","e3","e4","e5","e6","e7"):
            R[f[:-5]] = json.load(open(f"{OUT}/{f}"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    e1 = R.get("e1_scalability")
    if e1:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        xs = [r["routes"] for r in e1]
        ax[0].loglog(xs, [r["total_execution_time_s"] for r in e1], "o-", label="total time (s)")
        ax[0].loglog(xs, [r["avg_replan_time_ms"] / 1000 for r in e1], "s-", label="avg re-plan (s)")
        ax[0].set_xlabel("routes |E|"); ax[0].set_title("Runtime scaling"); ax[0].legend()
        ax[1].loglog(xs, [r["peak_memory_mb"] for r in e1], "o-"); ax[1].set_title("Peak memory (MB)"); ax[1].set_xlabel("routes |E|")
        ax[2].semilogx(xs, [r["search_space_reduction_pct"] for r in e1], "o-"); ax[2].set_title("Search-space reduction (%)"); ax[2].set_xlabel("routes |E|")
        for a in ax:
            a.grid(True, which="both", alpha=.3)
        plt.tight_layout(); plt.savefig(f"{OUT}/scalability.png", dpi=110); plt.close()
    e3 = R.get("e3_routing")
    if e3:
        fig, ax = plt.subplots(figsize=(6, 4))
        w = 0.35
        xs = range(len(e3))
        ax.bar([x - w / 2 for x in xs], [r["dijkstra_nodes_per_query"] for r in e3], w, label="Dijkstra")
        ax.bar([x + w / 2 for x in xs], [r["alt_nodes_per_query"] for r in e3], w, label="A* + ALT")
        ax.set_xticks(list(xs)); ax.set_xticklabels([r["dataset"] for r in e3]); ax.set_yscale("log")
        ax.set_ylabel("nodes explored / query"); ax.legend(); ax.set_title("Routing search effort")
        plt.tight_layout(); plt.savefig(f"{OUT}/routing_nodes.png", dpi=110); plt.close()
    e4 = R.get("e4_scheduling")
    if e4:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        h = e4["history"].get("ga_ls", {})
        if h.get("ga"):
            ax[0].plot(h["ga"], "o-", label="GA best / generation")
        if h.get("ls"):
            off = len(h.get("ga", [])) - 1
            ax[0].plot([off + i for i in range(len(h["ls"]))], h["ls"], label="Local search (hybrid stage)")
        ax[0].set_title("Hybrid convergence (objective)"); ax[0].set_xlabel("step"); ax[0].legend(); ax[0].grid(alpha=.3)
        rows = e4["rows"]
        ax[1].bar([r["method"] for r in rows], [r["gain_vs_greedy_%"] for r in rows]); ax[1].set_title("Gain over greedy (%)")
        plt.tight_layout(); plt.savefig(f"{OUT}/scheduling.png", dpi=110); plt.close()
    e2 = R.get("e2_baseline")
    if e2:
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ds = sorted({r["dataset"] for r in e2}, key=lambda d: DATASETS[d][0])
        for j, (key, title) in enumerate((("time_s", "Execution time (s)"), ("states", "States explored"), ("replan_ms", "Avg re-plan time (ms)"))):
            for i, cfgname in enumerate(("naive baseline", "Chronos (ours)")):
                vals = [next(r[key] for r in e2 if r["dataset"] == d and r["config"] == cfgname) for d in ds]
                ax[j].bar([k + i * 0.38 for k in range(len(ds))], vals, 0.38, label=cfgname)
            ax[j].set_xticks([k + 0.19 for k in range(len(ds))]); ax[j].set_xticklabels(ds); ax[j].set_yscale("log"); ax[j].set_title(title)
        ax[0].legend()
        plt.tight_layout(); plt.savefig(f"{OUT}/baseline_vs_ours.png", dpi=110); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--xl", action="store_true")
    ap.add_argument("--only", default="", help="comma list of e1..e7 (default: all)")
    ap.add_argument("--e2-medium", action="store_true", help="include the (slow) naive baseline on medium")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else {"e1", "e2", "e3", "e4", "e5", "e6", "e7"}
    seeds = [1, 2, 3]
    ds_small = ["tiny", "small", "medium"]
    if "e1" in only:
        print("E1 scalability", flush=True)
        e1_scalability(ds_small + (["large"] if a.large else []), {"xl(30k)": (30000, 300000, 15000)} if a.xl else {})
    if "e2" in only:
        print("E2 baseline vs ours", flush=True)
        e2_baseline(["small"] + (["medium"] if a.e2_medium else []), seeds[:2] if not a.quick else seeds[:1])
    if "e3" in only:
        print("E3 routing", flush=True); e3_routing(["small", "medium"] + (["large"] if a.large else []))
    if "e4" in only:
        print("E4 scheduling", flush=True); e4_scheduling("small", 30.0, seeds)
    if "e5" in only:
        print("E5 knapsack", flush=True); e5_infra()
    if "e6" in only:
        print("E6 flow", flush=True); e6_flow("medium")
    if "e7" in only:
        print("E7 dynamic", flush=True); e7_dynamic("medium", seeds[:2])
    plots()
    print("done")


if __name__ == "__main__":
    main()