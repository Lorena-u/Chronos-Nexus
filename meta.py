"""Meta-optimization layer: picks algorithms from problem size, urgency and budget."""
from .galaxy import PID


class Meta:
    def __init__(self, cfg, st):
        self.cfg, self.st = cfg, st

    def note(self, msg):
        self.st.log.append("[meta] " + msg)

    def routing(self, n_nodes):
        a = self.cfg.route_algo
        if a != "auto":
            return a
        return "astar-ALT" if n_nodes >= 300 else "dijkstra"

    def scheduling(self, n_missions, budget_s):
        m = self.cfg.sched_mode
        if m != "auto":
            self.note(f"scheduling forced -> {m}")
            return m
        # plan_all() further degrades GA -> LS -> greedy using the *measured* cost of one evaluation
        choice = "ga_ls" if n_missions <= 3000 else "greedy"
        self.note(f"scheduling: {n_missions} missions, budget {budget_s:.1f}s -> request {choice} "
                  f"(degrades to LS/greedy if evaluations are unaffordable)")
        return choice

    def infra(self, n_items):
        m = self.cfg.infra_mode
        choice = m if m != "auto" else "core"
        self.note(f"infrastructure: {n_items} candidates -> {choice} (LP fixing, then DP or B&B on the core)")
        return choice

    def replan(self, affected, total):
        if self.cfg.replan == "full":
            return "full"
        if total and affected / total > 0.3:
            self.note(f"event affects {affected}/{total} assignments (>30%) -> full re-plan of the future")
            return "full"
        return "incremental"