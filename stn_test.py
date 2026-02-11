from stn_opt_alg1 import *
# test code
# Step 1
# constraints = [
#     # Process P1 duration: Xe - Xs = 5  --> 5 <= Xe - Xs <= 5
#     STNConstraint("P1_s", "P1_e", LB=5, UB=5),

#     # Process P2 duration: 3
#     STNConstraint("P2_s", "P2_e", LB=3, UB=3),

#     # Precedence: P1 ends before P2 starts => P2_s - P1_e >= 0 (no upper bound -> use a large UB)
#     STNConstraint("P1_e", "P2_s", LB=0, UB=1e9),
# ]

# nodes, edges = build_distance_graph(constraints)
# nodes, edges = ensure_reference_node(nodes, edges, "X0")

# d = all_pairs_shortest_paths(nodes, edges)  # raises InfeasibleSTN if infeasible
# print("Feasible STN. Example distance d(P1_s, P2_s) =", d["P1_s"]["P2_s"])

# # Step 2
# landmarks = [10.0, 20.0]
# prices = [0.10, 0.30, 0.15]  # (-inf,10], (10,20], (20,inf)
# intervals = build_price_intervals(landmarks, prices)

# print(intervals)
# print(price_at(intervals, 9.0), price_at(intervals, 12.0), price_at(intervals, 100.0))


# Example processes (must match your STN node names)
processes = [
    Process(name="P1", start="P1_s", end="P1_e", energy=2.0),
    Process(name="P2", start="P2_s", end="P2_e", energy=1.5),
]

# Example intervals
intervals = build_price_intervals([10.0, 20.0], [0.10, 0.30, 0.15])

choices = build_activation_choices(processes, intervals)

for c in choices:
    print(c)