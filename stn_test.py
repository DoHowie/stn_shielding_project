from aaai18_alg1_1 import *
from aaai18_alg2 import *

constraints = [
    # Durations
    STNConstraint("S1", "E1", 4, 4), # P1 duration 4
    STNConstraint("S2", "E2", 3, 3), # P2 duration 3

    # Dependency: P2 starts after P1 ends
    STNConstraint("E1", "S2", 0, float("inf"))
]

processes = [
    Process("P1", "S1", "E1", 3), # P1 energy 3
    Process("P2", "S2", "E2", 2), # P2 energy 2
]

# Price intervals:
# I1 = (-inf, 5]  price 10
# I2 = (5, 10]    price 2
# I3 = (10, inf)  price 1
landmarks = [5, 10]
prices = [10, 2, 1]

schedule, makespan_T, cost = solve_alg2_cost_time_tradeoff(
    constraints,
    processes,
    landmarks,
    prices,
    gamma=2.0,      # allow up to 2x minimum cost
    T_low=0.0,
    T_high=50.0,
)

print("\n=== Algorithm 2 Result (2-process demo) ===")
print("Makespan T:", makespan_T)
print("Cost:", cost)
print("Schedule:")
for node, time in schedule.items():
    print(node, "=", time)