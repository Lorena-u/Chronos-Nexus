from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 1
    horizon_days: int = 30
    time_budget_s: float = 60.0      # soft computational budget for the planning phases
    route_algo: str = "auto"         # auto | dijkstra | astar
    sched_mode: str = "auto"         # auto | greedy | greedy_ls | ga_ls
    flow_mode: str = "mcmf"          # mcmf | greedy
    flow_k: int = 6                  # candidate suppliers per demand (search-space pruning)
    infra_mode: str = "auto"         # auto | greedy | dp | bnb | core
    replan: str = "incremental"      # incremental | full
    ship_k: int = 3                  # candidate spacecraft evaluated per mission
    landmarks: int = 4
    lookahead_days: int = 14         # proactive horizon (reactive baseline uses 1)
    pareto: bool = True              # multi-objective routing for emergency missions
    ga_pop: int = 24
    ga_gens: int = 40
    verbose: bool = False