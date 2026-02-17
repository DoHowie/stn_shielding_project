"""
This algorithm solve STN load scheduling under piecewise-constant prices
1. Build STN distance graph D(S)
2. Compute all-pairs shortest paths d(·,·)
3. Build POSET relation over (process, interval) activations
4. Build DAG G_Λ (nodes Y, edges, weights)
5. Build bipartite graph B and compute maximum-weight independent set via max-flow/min-cut
6. Activate the selected tuples in D'(S)
7. Compute and return the final schedule τ*(Xs_Pi) = -d'(Xs_Pi, X0), return the optimal start time for each process Pi.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Set
from collections import deque

# STN structures and algorithms

Node = str


@dataclass(frozen=True)
class STNConstraint:
    "Represents a simple temporal constraint: LB <= Xj - Xi <= UB"
    Xi: Node
    Xj: Node
    LB: float
    UB: float


@dataclass(frozen=True)
class Edge:
    "A directed edge u -> v with weight w in a distance graph, representing the difference constraint: Xv <= Xu + w"
    u: Node
    v: Node
    w: float


class InfeasibleSTN(Exception):
    pass


def build_distance_graph(constraints: Iterable[STNConstraint]) -> Tuple[List[Node], List[Edge]]:
    """
    Input: STNConstraint
    Output: distance graph D(S) = (X, E_D)
    For each constraint LB <= Xj - Xi <= UB:
      - Add edge Xi -> Xj with weight UB
      - Add edge Xj -> Xi with weight -LB
    """
    nodes_set: Set[Node] = set()
    edges: List[Edge] = []

    for c in constraints:
        nodes_set.add(c.Xi)
        nodes_set.add(c.Xj)

        # Upper bound
        edges.append(Edge(c.Xi, c.Xj, c.UB))

        # Lower bound
        edges.append(Edge(c.Xj, c.Xi, -c.LB))

    return sorted(nodes_set), edges


def ensure_reference_node(nodes: List[Node], edges: List[Edge], x0: Node = "X0") -> Tuple[List[Node], List[Edge]]:
    if x0 not in nodes:
        nodes = sorted(nodes + [x0])
    return nodes, edges


# Shortest paths / feasibility

def bellman_ford(nodes: List[Node], edges: List[Edge], source: Node) -> Dict[Node, float]:
    """
    Single-source shortest paths in a directed weighted graph.
    Raises InfeasibleSTN if a negative cycle is reachable from `source`.
    """
    INF = float("inf")
    dist: Dict[Node, float] = {n: INF for n in nodes}
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

    # Detect negative cycle reachable from source
    for e in edges:
        if dist[e.u] != INF and dist[e.u] + e.w < dist[e.v]:
            raise InfeasibleSTN(f"Negative cycle detected (reachable from {source}).")

    return dist


def all_pairs_shortest_paths(nodes: List[Node], edges: List[Edge]) -> Dict[Node, Dict[Node, float]]:
    "Returns d[u][v] = shortest path length from u to v in the distance graph."
    d: Dict[Node, Dict[Node, float]] = {}
    for s in nodes:
        d[s] = bellman_ford(nodes, edges, s)
    return d


# Price intervals (piecewise-constant f(t))

@dataclass(frozen=True)
class Interval:
    """
    Represents a price interval (left, right] with constant price.
    Use None for -inf (left) or +inf (right).
    """
    left: Optional[float]   # None means -inf
    right: Optional[float]  # None means +inf
    price: float            # constant price within this interval


def build_price_intervals(landmarks: List[float], prices: List[float]) -> List[Interval]:
    """
    Build intervals I = {I1, ..., I_{L+1}} from landmarks L = {l1, ..., lL}.

    landmarks: [l1, l2, ..., lL] strictly increasing
    prices: length L+1, where prices[j] is the price of interval I_{j+1}

    Intervals:
      I1      = (-inf, l1]
      I2      = (l1, l2]
      ...
      I_{L}   = (l_{L-1}, l_L]
      I_{L+1} = (l_L, +inf)
    """
    if sorted(landmarks) != landmarks:
        raise ValueError("landmarks must be sorted increasing")
    if len(prices) != len(landmarks) + 1:
        raise ValueError("prices must have length len(landmarks)+1")

    intervals: List[Interval] = []
    # I1
    intervals.append(Interval(left=None, right=landmarks[0], price=prices[0]))
    # I2..I_L
    for i in range(1, len(landmarks)):
        intervals.append(Interval(left=landmarks[i - 1], right=landmarks[i], price=prices[i]))
    # I_{L+1}
    intervals.append(Interval(left=landmarks[-1], right=None, price=prices[-1]))
    return intervals


# Processes and activation tuples (Pi, Ij)

@dataclass(frozen=True)
class Process:
    """
    A process Pi with:
      - start time-point node Xs_Pi
      - end time-point node Xe_Pi
      - energy/load Wi
    """
    name: str
    start: Node
    end: Node
    energy: float


@dataclass(frozen=True)
class ActivationTuple:
    """
    Represents the POSET element ⟨Pi, Ij⟩.
    process Pi starts in interval Ij
    """
    proc_idx: int # index of Pi in the processes list
    interval_idx: int # index of Ij in the intervals list
    start_node: Node # Xs_Pi
    left: Optional[float] # left endpoint of interval Ij
    right: Optional[float] # right endpoint of interval Ij
    price: float # price f(t) on interval Ij


def build_activation_tuples(processes: List[Process], intervals: List[Interval]) -> List[ActivationTuple]:
    "Create all POSET elements ⟨Pi, Ij⟩ for all processes Pi and all intervals Ij."
    tuples: List[ActivationTuple] = []
    for i, p in enumerate(processes):
        for j, iv in enumerate(intervals):
            tuples.append(
                ActivationTuple(
                    proc_idx=i,
                    interval_idx=j,
                    start_node=p.start,
                    left=iv.left,
                    right=iv.right,
                    price=iv.price,
                )
            )
    return tuples


# POSET precedence relation, detecting binary conflicts between tuples

def poset_precedes(d: Dict[Node, Dict[Node, float]], a: ActivationTuple,  b: ActivationTuple,
) -> bool:
    """
      ⟨Pu, Ia⟩ ≺ ⟨Pv, Ib⟩  iff  l_b + d(Xs_Pv, Xs_Pu) - l_(a-1) <= 0
      it's <= 0 because we consider epsilon limit handling.

    In our data:
      l_b      = b.right   (right endpoint of interval Ib)
      l_(a-1)  = a.left    (left endpoint of interval Ia)

    If a.left is -inf or b.right is +inf, the expression cannot be <= 0 in the intended sense,
    so we return False (these “infinite endpoint” cases should not create a minimal binary conflict).

    here we are just checking if we picked pv with interval Ib, then whether pu with interval Ia would cause a conflict.
    conflict if the earilest time for process u > the latest time for process v + the time difference between their start nodes.   
    """
    # l_(a-1)
    if a.left is None: # -inf
        return False
    # l_b
    if b.right is None: # +inf
        return False

    return (b.right + d[b.start_node][a.start_node] - a.left) <= 0.0


# Build DAG G_Λ = (Y, edges) with node weights c_ij

def is_non_unary_conflict(d: Dict[Node, Dict[Node, float]], x0: Node, t: ActivationTuple,  # ⟨Pi, Ij⟩
) -> bool:
    """
    Include node y⟨Pi,Ij⟩ in Y iff:
      d(X0, Xs_Pi) - l_(j-1) > 0    AND    l_j + d(Xs_Pi, X0) >= 0
    """
    xs = t.start_node

    # d(X0, xs) - left > 0
    if t.left is None:  # left = -inf => always satisfied
        cond1 = True
    else:
        cond1 = (d[x0][xs] - t.left) > 0.0

    # right + d(xs, X0) >= 0
    if t.right is None:  # right = +inf => always satisfied
        cond2 = True
    else:
        cond2 = (t.right + d[xs][x0]) >= 0.0

    return cond1 and cond2


def compute_node_weight_cij(processes: List[Process], t: ActivationTuple, max_price: float,
) -> float:
    """
      c_ij = [ Σ_i' ( max_j' (Wi' * f(l_j')) ) ] - Wi*f(l_j) + 1

    Since f is piecewise constant over intervals, we implement the paper's intent by:
      max_j' f(l_j')  -> max interval price
      max_j' (Wi' * f(·)) -> Wi' * max_price

    Therefore:
      Σ_i' Wi' * max_price  - Wi * price_j + 1
    """
    sumW = sum(p.energy for p in processes)
    Wi = processes[t.proc_idx].energy
    return (sumW * max_price) - (Wi * t.price) + 1.0


def build_activation_dag(processes: List[Process], tuples_all: List[ActivationTuple], d: Dict[Node, Dict[Node, float]], x0: Node = "X0",
) -> Tuple[List[int], Dict[int, float], List[Tuple[int, int]]]:
    """
    Build node set Y (subset of activation tuples), weights c_ij, and edges of G_Λ.

    Returns:
      Y_ids: indices of tuples_all that are included in Y
      weight: mapping tuple_index -> c_ij
      edges: list of directed edges (src_tuple_index, dst_tuple_index) in G_Λ
    """
    # filter unary conflicts
    Y_ids: List[int] = [idx for idx, t in enumerate(tuples_all) if is_non_unary_conflict(d, x0, t)]

    # compute weights
    max_price = max((t.price for t in tuples_all), default=0.0)
    weight: Dict[int, float] = {}
    for idx in Y_ids:
        weight[idx] = compute_node_weight_cij(processes, tuples_all[idx], max_price)

    # build directed edges
    Y_set = set(Y_ids)
    edges: List[Tuple[int, int]] = []
    for a_idx in Y_ids:
        a = tuples_all[a_idx]
        for b_idx in Y_ids:
            if a_idx == b_idx:
                continue
            b = tuples_all[b_idx]
            # If a ≺ b then add edge y_b -> y_a
            if poset_precedes(d, a, b):
                edges.append((b_idx, a_idx))

    return Y_ids, weight, edges


# MWIS (Max-weight independent set) via bipartite min-cut

class Dinic:
    def __init__(self, n: int):
        self.n = n
        self.adj: List[List[List[float | int]]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: float) -> None:
        # forward edge: [to, cap, rev_index]
        self.adj[u].append([v, cap, len(self.adj[v])])
        # reverse edge
        self.adj[v].append([u, 0.0, len(self.adj[u]) - 1])

    def max_flow(self, s: int, t: int) -> float:
        flow = 0.0
        INF = 1e30

        while True:
            level = [-1] * self.n
            q = deque([s])
            level[s] = 0

            while q:
                u = q.popleft()
                for v, cap, rev in self.adj[u]:
                    if cap > 1e-12 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)

            if level[t] < 0:
                break

            it = [0] * self.n

            def dfs(u: int, f: float) -> float:
                if u == t:
                    return f
                for i in range(it[u], len(self.adj[u])):
                    it[u] = i
                    v, cap, rev = self.adj[u][i]
                    if cap > 1e-12 and level[v] == level[u] + 1:
                        pushed = dfs(v, min(f, cap))
                        if pushed > 1e-12:
                            # update forward
                            self.adj[u][i][1] -= pushed
                            # update reverse
                            self.adj[v][rev][1] += pushed
                            return pushed
                return 0.0

            while True:
                pushed = dfs(s, INF)
                if pushed <= 1e-12:
                    break
                flow += pushed

        return flow

    def reachable_from(self, s: int) -> List[bool]:
        """Nodes reachable from s in the residual graph (cap > 0)."""
        seen = [False] * self.n
        q = deque([s])
        seen[s] = True

        while q:
            u = q.popleft()
            for v, cap, rev in self.adj[u]:
                if cap > 1e-12 and not seen[v]:
                    seen[v] = True
                    q.append(v)
        return seen


def maximum_weight_independent_set_in_bipartite(Y_ids: List[int], weights: Dict[int, float], dag_edges: List[Tuple[int, int]],
) -> Set[int]:
    """
    Algorithm 1 lines 18-21:

    Line 18-19:
      Build bipartite graph B = (Y, Y', E) where Y' is a copy of Y.
      Add directed edge y⟨Pu,Ia⟩ -> y'⟨Pv,Ib⟩ iff G_Λ has edge y⟨Pu,Ia⟩ -> y⟨Pv,Ib⟩.

    Line 20:
      Compute maximum-weight independent set QB of B via maxflow/mincut.

    Line 21:
      Q_GΛ = { y⟨Pi,Ij⟩ : (y⟨Pi,Ij⟩ in QB) AND (y'⟨Pi,Ij⟩ in QB) }

    Returns:
      Q_GΛ as a set of tuple indices (subset of Y_ids).
    """
    m = len(Y_ids)
    # Compact indexing for left and right copies
    left_index = {tid: i for i, tid in enumerate(Y_ids)}
    right_index = {tid: i for i, tid in enumerate(Y_ids)}

    SRC = 2 * m
    SNK = 2 * m + 1
    dinic = Dinic(2 * m + 2)
    INF = 1e18

    # Build min-cut network for min-weight vertex cover in bipartite graphs:
    # SRC -> Left nodes (cap = weight)
    # Right nodes -> SNK (cap = weight)
    # Left -> Right edges (cap = INF)
    for tid in Y_ids:
        w = weights[tid]
        if w < 0:
            raise ValueError("Algorithm 1 expects positive node weights c_ij; got negative weight.")
        dinic.add_edge(SRC, left_index[tid], w)
        dinic.add_edge(m + right_index[tid], SNK, w)

    # Add bipartite edges based on DAG edges in G_Λ
    # If G_Λ has edge y_u -> y_v, then B has edge y_u -> y'_v.
    for u_tid, v_tid in dag_edges:
        if u_tid in left_index and v_tid in right_index:
            dinic.add_edge(left_index[u_tid], m + right_index[v_tid], INF)

    dinic.max_flow(SRC, SNK)
    reach = dinic.reachable_from(SRC)

    # Min vertex cover in bipartite graph from residual reachability:
    # cover = (Left \ Reachable) ∪ (Right ∩ Reachable)
    left_in_cover: Set[int] = set()
    right_in_cover: Set[int] = set()
    for tid in Y_ids:
        li = left_index[tid]
        ri = m + right_index[tid]
        if not reach[li]:
            left_in_cover.add(tid)
        if reach[ri]:
            right_in_cover.add(tid)

    # Independent set QB = complement of vertex cover:
    left_in_IS = set(Y_ids) - left_in_cover
    right_in_IS = set(Y_ids) - right_in_cover

    #Keep only nodes whose left and right copies are both in QB
    Q_G_lambda = left_in_IS.intersection(right_in_IS)
    return Q_G_lambda


# Activate selected tuples and compute τ*

def add_activation_edges_to_distance_graph(edges: List[Edge], x0: Node, t: ActivationTuple,
) -> List[Edge]:
    """
    For each selected y⟨Pi,Ij⟩, add:

      X0 -> Xs_Pi with weight l_j      (interval right endpoint)
      Xs_Pi -> X0 with weight -l_(j-1) (negative of interval left endpoint)

    If a boundary is infinite (None), we skip adding that edge.
    """
    extra: List[Edge] = []
    xs = t.start_node
    EPS = 1e-9  # tiny strictness surrogate

    # Add X0 -> Xs_Pi with l_j (right endpoint)
    if t.right is not None:
        extra.append(Edge(x0, xs, t.right))

    # Add Xs_Pi -> X0 with -l_(j-1) (left endpoint)
    if t.left is not None:
        extra.append(Edge(xs, x0, -(t.left + EPS)))

    return edges + extra


def compute_schedule_from_modified_graph(nodes: List[Node], edges: List[Edge], processes: List[Process], x0: Node = "X0",
) -> Dict[Node, float]:
    """
    Compute shortest path distances d'(Xs_Pi, X0) in D'(S) and output:
      τ*(Xs_Pi) = - d'(Xs_Pi, X0)

    Returns a dict mapping each process start node Xs_Pi to τ*(Xs_Pi).
    """
    schedule: Dict[Node, float] = {}
    for p in processes:
        dist = bellman_ford(nodes, edges, p.start)
        if dist[x0] == float("inf"):
            raise ValueError(f"X0 not reachable from {p.start} in modified graph; cannot compute τ*(Xs_Pi).")
        schedule[p.start] = -dist[x0]
    return schedule



# Main solver, the whole Alg1

def solve_stn_load_scheduling(stn_constraints: Iterable[STNConstraint], processes: List[Process], landmarks: List[float], prices: List[float], x0: Node = "X0",
) -> Dict[Node, float]:
    """
    Inputs:
      - stn_constraints: STN constraints E
      - processes: list of processes Pi with start/end nodes in the STN and energy Wi
      - landmarks, prices: piecewise-constant price function f(t)

    Output:
      - schedule τ*: dictionary mapping each process start node Xs_Pi to τ*(Xs_Pi)

    Note:
      - The algorithm assumes the base STN is consistent; if not, Bellman-Ford will raise InfeasibleSTN.
    """
    # distance graph D(S)
    nodes, D_edges = build_distance_graph(stn_constraints)
    nodes, D_edges = ensure_reference_node(nodes, D_edges, x0=x0)

    # all-pairs shortest paths in D(S)
    d = all_pairs_shortest_paths(nodes, D_edges)  # may raise InfeasibleSTN if STN infeasible

    # landmarks and intervals for f(t)
    intervals = build_price_intervals(landmarks, prices)

    # construct POSET elements and precedence relation
    tuples_all = build_activation_tuples(processes, intervals)

    # build DAG G_Λ (Y nodes, edges) and node weights c_ij
    Y_ids, weights, dag_edges = build_activation_dag(processes, tuples_all, d, x0=x0)

    # compute maximum-weight independent set Q_GΛ via bipartite maxflow/mincut
    Q_G_lambda_ids = maximum_weight_independent_set_in_bipartite(Y_ids, weights, dag_edges)

    # construct modified distance graph D'(S) by adding activation edges for Q_GΛ
    D_prime_edges = list(D_edges)
    for tid in Q_G_lambda_ids:
        D_prime_edges = add_activation_edges_to_distance_graph(D_prime_edges, x0, tuples_all[tid])

    # compute and return τ*(Xs_Pi) = -d'(Xs_Pi, X0)
    return compute_schedule_from_modified_graph(nodes, D_prime_edges, processes, x0=x0)
