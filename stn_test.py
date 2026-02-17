from aaai18_alg1_1 import *
from aaai18_alg2 import *


constraints = [
    # Durations
    STNConstraint("S1", "E1", 4, 4),
    STNConstraint("S2", "E2", 3, 3),
    STNConstraint("S3", "E3", 5, 5),

    # Sequential dependencies
    STNConstraint("E1", "S2", 0, float("inf")),
    STNConstraint("E2", "S3", 0, float("inf")),
]

processes = [
    Process("P1", "S1", "E1", 3),
    Process("P2", "S2", "E2", 2),
    Process("P3", "S3", "E3", 4),
]

landmarks = [5, 15]
prices = [10, 2, 1]  # expensive, medium, cheap


schedule, makespan_T, cost = solve_alg2_cost_time_tradeoff(
    constraints,
    processes,
    landmarks,
    prices,
    gamma=2.0,        # allow cost up to 2x optimal
    T_low=0.0,
    T_high=100.0,
)

print("\n=== Algorithm 2 Result ===")
print("Makespan T:", makespan_T)
print("Cost:", cost)
print("Schedule:")
for node, time in schedule.items():
    print(node, "=", time)
