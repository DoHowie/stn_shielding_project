from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from aaai18_alg1_1 import *

def schedule_cost(schedule: Dict[str, float], processes: List[Process], landmarks: List[float], prices: List[float]) -> float:
    """
    Input: schedule mapping process start nodes to start times, list of processes, price intervals
    Output: total cost of the schedule, computed by summing energy * price for each process
    """
    intervals = build_price_intervals(landmarks, prices)
    total = 0.0
    for p in processes:
        t = schedule[p.start]
        for iv in intervals:
            left_ok = (iv.left is None) or (t > iv.left)
            right_ok = (iv.right is None) or (t <= iv.right)
            if left_ok and right_ok:
                total += p.energy * iv.price
                break
        else:
            raise RuntimeError(f"Start time {t} for {p.name} not in any interval.")
    return total


def solve_alg2_cost_time_tradeoff(
    base_constraints: List[STNConstraint],
    processes: List[Process],
    landmarks: List[float],
    prices: List[float],
    gamma: float,
    x0: str = "X0",
    # binary search configuration
    T_low: float = 0.0,
    T_high: float = 1_000.0,
    iters: int = 40,
) -> Tuple[Dict[str, float], float, float]:
    """
    Algorithm 2 (high-level):
      1) run Alg1 → get c*
      2) binary search makespan T:
           add constraints Xe_i - X0 <= T  for all processes
           run Alg1 → get schedule and cost
           accept if cost <= gamma*c*
      Returns:
        (best_schedule, best_makespan_T, best_cost)
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1.0")

    # Step 1: get minimum cost schedule
    best_cost_schedule = solve_stn_load_scheduling(
        base_constraints, processes, landmarks, prices, x0=x0
    )
    c_star = schedule_cost(best_cost_schedule, processes, landmarks, prices)

    # Helper: try a makespan T by adding deadline constraints and re-running Alg1
    def try_T(T: float) -> Optional[Tuple[Dict[str, float], float]]:
        extra = []
        for p in processes:
            # Xe_i - X0 <= T  <=>  Xe_i <= X0 + T
            # This is STNConstraint(X0, Xe_i, LB=-inf, UB=T) but we use a large negative LB.
            extra.append(STNConstraint(x0, p.end, LB=-1e18, UB=T))
        constraints_T = base_constraints + extra

        try:
            sched = solve_stn_load_scheduling(constraints_T, processes, landmarks, prices, x0=x0)
        except Exception:
            return None

        cost = schedule_cost(sched, processes, landmarks, prices)
        return sched, cost

    # Binary search
    lo, hi = T_low, T_high
    best: Optional[Tuple[Dict[str, float], float, float]] = None

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res = try_T(mid)
        if res is None:
            # infeasible STN under this makespan
            lo = mid
            continue

        sched_mid, cost_mid = res
        if cost_mid <= gamma * c_star:
            # feasible within budget → try smaller makespan
            best = (sched_mid, mid, cost_mid)
            hi = mid
        else:
            # too expensive → need more slack time
            lo = mid

    if best is None:
        raise RuntimeError("No feasible makespan found within [T_low, T_high]. Try increasing T_high.")

    return best
