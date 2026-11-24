"""Chronos Nexus CLI.   python main.py --dataset small --seed 1 [--baseline] [--json out.json]"""
import argparse
import json

from chronos.config import Config
from chronos.generator import DATASETS
from chronos.sim import Simulation

def baseline_cfg(cfg):
    """Naive configuration used in the experiments: plain Dijkstra, scan all ships, no pruning, full re-plan."""
    cfg.route_algo, cfg.ship_k, cfg.flow_k = "dijkstra", 10 ** 9, 60
    cfg.sched_mode, cfg.infra_mode, cfg.replan, cfg.pareto = "greedy", "dp", "full", False
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="small", choices=list(DATASETS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=60.0)
    ap.add_argument("--baseline", action="store_true", help="naive (no pruning / no incremental) configuration")
    ap.add_argument("--sched", default=None, choices=["auto", "greedy", "greedy_ls", "ga_ls"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cfg = Config(seed=a.seed, time_budget_s=a.time_budget)
    if a.baseline:
        baseline_cfg(cfg)
    if a.sched:
        cfg.sched_mode = a.sched
    rep = Simulation(cfg, a.dataset).run()
    keys = ["civilization_score", "population_survival_rate", "resources_delivered_units", "shortage_units",
            "resource_waste_units", "missions_completed", "missions_failed", "total_distance", "total_fuel",
            "total_transport_cost", "total_risk_expected_failures", "graph_nodes_explored",
            "candidate_solutions_evaluated", "states_explored", "search_space_reduction_pct",
            "events_handled", "avg_replan_time_ms", "total_execution_time_s", "peak_memory_mb"]
    print(f"=== Chronos Nexus | {a.dataset} | {'BASELINE' if a.baseline else 'OPTIMIZED'} ===")
    for k in keys:
        print(f"{k:34s} {rep[k]}")
    print("components:", rep["components"])
    print("reduction breakdown:", rep["reduction_breakdown_pct"])
    print("infrastructure:", rep["infrastructure"])
    print("predicted shortages at t0:", rep["predicted_shortages_at_t0"], rep["predicted_examples"])
    for line in rep["decision_log"]:
        print("  ", line)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()