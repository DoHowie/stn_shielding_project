from pprint import pprint

from aaai18_alg1_1 import *


def build_figure2_instance():
    """
    Nodes:
      X0 = reference timepoint (12:00am)
      Xh_s, Xh_e = heater start/end
      Xc_s, Xc_e = cooker start/end
      Xd_s, Xd_e = dishwasher start/end

    STN constraints read directly from Figure 2.
    """
    constraints = [
        # Heater
        STNConstraint("X0",   "Xh_s", 420, 900),
        STNConstraint("Xh_s", "Xh_e",  90,  90),

        # Cooker
        STNConstraint("X0",   "Xc_s",   0, float("inf")),
        STNConstraint("Xc_s", "Xc_e",  30,  30),
        STNConstraint("X0",   "Xc_e", 690, 1050),

        # Dishwasher
        STNConstraint("X0",   "Xd_s", 1080, 1380),
        STNConstraint("Xd_s", "Xd_e",   60,   60),

        # Cross-process constraints
        STNConstraint("Xh_e", "Xc_s",   0, float("inf")),
        STNConstraint("Xh_e", "Xd_s",   0, 600),
        STNConstraint("Xc_e", "Xd_s",   0, float("inf")),
    ]

    processes = [
        Process("Cooker",     start="Xc_s", end="Xc_e", energy=0.2),
        Process("Heater",     start="Xh_s", end="Xh_e", energy=1.3),
        Process("Dishwasher", start="Xd_s", end="Xd_e", energy=0.5),
    ]

    landmarks = [480, 840, 1200, 1320]
    prices = [0.13, 0.28, 0.45, 0.28, 0.13]

    return constraints, processes, landmarks, prices


def summarize_selected_tuples(selected_ids, tuples_all, processes):
    summary = []
    for tid in sorted(selected_ids):
        t = tuples_all[tid]
        proc = processes[t.proc_idx].name
        interval_name = f"I{t.interval_idx + 1}"
        summary.append((proc, interval_name, t.left, t.right, t.price))
    return summary


def format_distance_matrix(d):
    """
    Input: d, the all-pairs shortest path matrix from the distance graph
    Output: (cols, matrix) where cols is the list of column names and matrix is a list of (row_name, row_values) for printing.
    """
    rows = ["X0", "Xc_s", "Xh_s", "Xd_s"]
    cols = ["X0", "Xc_s", "Xh_s", "Xd_s"]
    matrix = []
    for r in rows:
        row = []
        for c in cols:
            val = d[r][c]
            if val == float("inf"):
                row.append("inf")
            else:
                row.append(int(val))
        matrix.append((r, row))
    return cols, matrix


def main():
    constraints, processes, landmarks, prices = build_figure2_instance()
    nodes, edges = build_distance_graph(constraints)
    nodes = ensure_reference_node(nodes, x0="X0")
    d = all_pairs_shortest_paths(nodes, edges)

    expected = {
        "X0":   {"X0": 0,    "Xc_s": 1020, "Xh_s": 900,  "Xd_s": 1380},
        "Xc_s": {"X0": -660, "Xc_s": 0,    "Xh_s": -90,  "Xd_s": 600},
        "Xh_s": {"X0": -420, "Xc_s": 600,  "Xh_s": 0,    "Xd_s": 690},
        "Xd_s": {"X0": -1080,"Xc_s": -60,  "Xh_s": -180, "Xd_s": 0},
    }

    matrix_ok = True
    for r in expected:
        for c in expected[r]:
            got = d[r][c]
            if abs(got - expected[r][c]) > 1e-9:
                matrix_ok = False
                print(f"MISMATCH in shortest paths: d[{r}][{c}] = {got}, expected {expected[r][c]}")

    print("=" * 80)
    print("Figure 2 shortest-path matrix check")
    cols, matrix = format_distance_matrix(d)
    print("Columns:", cols)
    for r, row in matrix:
        print(f"{r:4s}: {row}")
    print("Matches paper Figure 2 matrix:", matrix_ok)

    intervals = build_price_intervals(landmarks, prices)
    tuples_all = build_activation_tuples(processes, intervals)
    Y_ids, weights, dag_edges = build_activation_dag(processes, tuples_all, d, x0="X0")
    selected_ids = maximum_weight_independent_set_in_bipartite(Y_ids, weights, dag_edges)
    selected_summary = summarize_selected_tuples(selected_ids, tuples_all, processes)

    print("\n" + "=" * 80)
    pprint(selected_summary)

    expected_selection = {("Cooker", "I2"), ("Heater", "I1"), ("Dishwasher", "I3")}
    got_selection = {(p, i) for p, i, *_ in selected_summary}
    print("Matches paper Figure 5 MWIS:", got_selection == expected_selection)

    d_prime_edges = list(edges)
    for tid in selected_ids:
        d_prime_edges = add_activation_edges_to_distance_graph(d_prime_edges, "X0", tuples_all[tid])

    schedule = compute_schedule_from_modified_graph(nodes, d_prime_edges, processes, x0="X0")

    print("\n" + "=" * 80)
    print("Recovered start-time schedule τ*(Xs_Pi)")
    for p in processes:
        print(f"{p.name:12s} ({p.start}): {schedule[p.start]:.6f} minutes")

    def price_at(t):
        if t <= 480:
            return 0.13
        elif t <= 840:
            return 0.28
        elif t <= 1200:
            return 0.45
        elif t <= 1320:
            return 0.28
        else:
            return 0.13

    total_cost = sum(price_at(schedule[p.start]) * p.energy for p in processes)
    print(f"Total cost of recovered schedule: {total_cost:.6f}")

    print("\nInterpretation:")
    print("- Heater starts in I1 = (-inf, 480]")
    print("- Cooker starts in I2 = (480, 840]")
    print("- Dishwasher starts in I3 = (840, 1200]")
    print("This should agree with the paper's Figure 5 MWIS result.")


if __name__ == "__main__":
    main()
