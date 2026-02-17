from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set

Node = str

@dataclass(frozen=True)
class STNConstraint:
    Xi: Node
    Xj: Node
    LB: float
    UB: float

@dataclass(frozen=True)
class Edge:
    u: Node
    v: Node
    w: float

@dataclass(frozen=True)
class Interval:
    left: Optional[float]   # None means -inf
    right: Optional[float]  # None means +inf
    price: float

class InfeasibleSTN(Exception):
    pass

def build_distance_graph(constraints: Iterable[STNConstraint]) -> Tuple[List[Node], List[Edge]]:
    """
    Convert a list of STN constraints into a distance graph.
    Input: list of STN constraints
    Output: list of nodes and list of edges
    """
    nodes_set = set()
    edges: List[Edge] = []
    for c in constraints:
        nodes_set.add(c.Xi)
        nodes_set.add(c.Xj)
        # Upper bound
        edges.append(Edge(c.Xi, c.Xj, c.UB))
        # Lower bound
        edges.append(Edge(c.Xj, c.Xi, -c.LB))
    nodes = sorted(nodes_set)
    return nodes, edges

def bellman_ford(nodes: List[Node], edges: List[Edge], source: Node) -> Dict[Node, float]:
    """
    Single-source shortest paths; raises InfeasibleSTN if a negative cycle is reachable.
    Input: nodes, edges, and a source node
    Output: a dictionary containing the shortest paths from the chosen source node
    """
    INF = float("inf")
    dist = {n: INF for n in nodes}
    dist[source] = 0.0

    # Relax edges |V|-1 times
    for _ in range(len(nodes) - 1):
        updated = False
        for e in edges:
            if dist[e.u] != INF and dist[e.u] + e.w < dist[e.v]:
                dist[e.v] = dist[e.u] + e.w
                updated = True
        if not updated:
            break

    # Detect negative cycle
    for e in edges:
        if dist[e.u] != INF and dist[e.u] + e.w < dist[e.v]:
            raise InfeasibleSTN(f"Negative cycle detected (reachable from {source}).")

    return dist

def all_pairs_shortest_paths(nodes: List[Node], edges: List[Edge]) -> Dict[Node, Dict[Node, float]]:
    """
    All-pairs shortest paths via Bellman-Ford from each node.
    Input: nodes and edges
    Output: a dictionary containing shortest paths for all nodes
    """
    d: Dict[Node, Dict[Node, float]] = {}
    for s in nodes:
        d[s] = bellman_ford(nodes, edges, s)
    return d

def ensure_reference_node(nodes: List[Node], edges: List[Edge], x0: Node = "X0") -> Tuple[List[Node], List[Edge]]:
    """
    Helper function to ensure X0 is present.
    Input: nodes, edges, and x0 node
    Output: new nodes, new edges containing x0
    """
    if x0 not in nodes:
        nodes = sorted(nodes + [x0])
    return nodes, edges

def build_price_intervals(landmarks: List[float], prices: List[float]) -> List[Interval]:
    """
    landmarks: [l1, l2, ..., lL] strictly increasing
    prices: length L+1, prices[j] is the price on interval I_{j+1}
    Input: list of landmarks, and a list of prices for each landmarks
    Output: list of intervals
    """
    if sorted(landmarks) != landmarks:
        raise ValueError("landmarks must be sorted increasing")
    if len(prices) != len(landmarks) + 1:
        raise ValueError("prices must have length len(landmarks)+1")

    intervals: List[Interval] = []
    # I1
    intervals.append(Interval(left=None, right=landmarks[0], price=prices[0]))
    # I2..IL
    for i in range(1, len(landmarks)):
        intervals.append(Interval(left=landmarks[i-1], right=landmarks[i], price=prices[i]))
    # I_{L+1}
    intervals.append(Interval(left=landmarks[-1], right=None, price=prices[-1]))
    return intervals

def price_at(intervals: List[Interval], t: float) -> float:
    """Evaluate the piecewise-constant price at time t."""
    for iv in intervals:
        left_ok = True if iv.left is None else (t > iv.left)
        right_ok = True if iv.right is None else (t <= iv.right)
        if left_ok and right_ok:
            return iv.price
    raise RuntimeError("Time t did not fall into any interval (bug).")

@dataclass(frozen=True)
class Process:
    name: str
    start: Node
    end: Node
    energy: float

@dataclass(frozen=True)
class ActivationChoice:
    """
    Represents lambda_{i,j}: 'process i starts in interval j'
    """
    proc_idx: int          # process i
    interval_idx: int      # interval j
    start_node: Node       # X_i^s (we add interval edges to/from X0 and this node later)
    left: Optional[float]  # interval left boundary (l_{j-1}), None means -inf
    right: Optional[float] # interval right boundary (l_j), None means +inf
    price: float           # f(I_j)
    cost: float            # W_i * f(I_j)


def build_activation_choices(processes: List[Process], intervals: List[Interval]) -> List[ActivationChoice]:
    """
    For each process P_i and each price interval I_j, create lambda_{i,j} with:
      cost_{i,j} = W_i * price(I_j)
    """
    choices: List[ActivationChoice] = []

    for i, p in enumerate(processes):
        for j, iv in enumerate(intervals):
            cost = p.energy * iv.price
            choices.append(
                ActivationChoice(
                    proc_idx=i,
                    interval_idx=j,
                    start_node=p.start,
                    left=iv.left,
                    right=iv.right,
                    price=iv.price,
                    cost=cost,
                )
            )

    return choices

def is_stn_feasible(nodes: List[Node], edges: List[Edge]) -> bool:
    """
    Return True iff the STN (difference constraints graph) has no negative cycle.

    Trick: add a super-source 'SS' with 0-weight edges to all nodes, then run
    Bellman-Ford once. If any negative cycle exists anywhere, it will be reachable.
    """
    super_source = "__SS__"
    if super_source in nodes:
        raise ValueError("Super-source name collision. Rename __SS__ to something unique.")

    nodes2 = nodes + [super_source]
    edges2 = edges + [Edge(super_source, n, 0.0) for n in nodes]

    try:
        _ = bellman_ford(nodes2, edges2, super_source)
        return True
    except InfeasibleSTN:
        return False

def interval_edges_for_choice(choice: ActivationChoice, x0: Node = "X0") -> List[Edge]:
    """
    Given a choice 'process starts in (left, right]', return the STN edges that enforce it.

    - Enforce X_start <= right  via edge (X0 -> X_start, right)
    - Enforce X_start >= left   via edge (X_start -> X0, -left)

    If left is None, no lower bound.
    If right is None, no upper bound.
    """
    xs = choice.start_node
    extra: List[Edge] = []

    if choice.right is not None:
        extra.append(Edge(x0, xs, choice.right))

    if choice.left is not None:
        extra.append(Edge(xs, x0, -choice.left))

    return extra

def filter_unary_feasible_choices(nodes: List[Node], base_edges: List[Edge], choices: List[ActivationChoice], x0: Node = "X0",
) -> List[ActivationChoice]:
    """
    Step 5: Unary conflicts check.
    Keep only those choices lambda_{i,j} that remain feasible when their interval bounds are added.
    """
    kept: List[ActivationChoice] = []

    for ch in choices:
        extra_edges = interval_edges_for_choice(ch, x0=x0)
        if is_stn_feasible(nodes, base_edges + extra_edges):
            kept.append(ch)

    return kept


def build_binary_conflicts(nodes: List[Node], base_edges: List[Edge], choices: List[ActivationChoice], x0: Node = "X0",
) -> Dict[int, Set[int]]:
    """
    Step 6: Binary conflicts.
    conflict[a] contains b iff (choice a AND choice b) together makes STN infeasible.
    We store conflicts by indices in the `choices` list.

    NOTE: We skip pairs from the same process because Step 7 will enforce "choose 1 per process" anyway.
    """
    conflict: Dict[int, Set[int]] = {i: set() for i in range(len(choices))}

    # Precompute interval edges for each choice so we don't rebuild them repeatedly
    choice_edges: List[List[Edge]] = [interval_edges_for_choice(ch, x0=x0) for ch in choices]

    for i in range(len(choices)):
        for j in range(i + 1, len(choices)):
            # same process: mutually exclusive by definition, not a feasibility conflict
            if choices[i].proc_idx == choices[j].proc_idx:
                continue

            edges_ij = base_edges + choice_edges[i] + choice_edges[j]
            if not is_stn_feasible(nodes, edges_ij):
                conflict[i].add(j)
                conflict[j].add(i)

    return conflict


