"""
This algorithm solve STN load scheduling under piecewise-constant prices
1. Build STN distance graph D(S), contains all nodes and edges representing temporal constraints
2. Compute all-pairs shortest paths d(·,·)
3. Build POSET relation over (process, interval) activations
4. Build DAG G (nodes Y, edges, weights)
5. Build bipartite graph B and compute maximum-weight independent set via max-flow/min-cut
6. Activate the selected tuples in D'(S)
7. Compute and return the final schedule τ*(Xs_Pi) = -d'(Xs_Pi, X0), return the optimal start time for each process Pi.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Set
from collections import deque

Node = str

@dataclass(frozen=True)
class STNConstraint:
    "Represents a simple temporal constraint: LB <= Xj - Xi <= UB"
    "In other words, the time difference between Xj and Xi must be between LB and UB."
    "event Xi must occur at least LB time before Xj, and at most UB time before Xj"
    Xi: Node
    Xj: Node
    LB: float
    UB: float

    def __post_init__(self):
        if self.LB > self.UB:
            raise ValueError(
                f"Invalid STN constraint: LB ({self.LB}) > UB ({self.UB}) "
                f"for constraint {self.Xi} -> {self.Xj}"
            )


@dataclass(frozen=True)
class Edge:
    "A directed edge u -> v with weight w in a distance graph, representing the difference constraint: Xv <= Xu + w"
    "Xv must occur at most w time after Xu"
    u: Node
    v: Node
    w: float


class InfeasibleSTN(Exception):
    pass


def build_distance_graph(constraints: Iterable[STNConstraint]) -> Tuple[List[Node], List[Edge]]:
    """
    Input: a list of STNConstraint
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
        edges.append(Edge(c.Xi, c.Xj, c.UB))
        edges.append(Edge(c.Xj, c.Xi, -c.LB))

    return sorted(nodes_set), edges


def ensure_reference_node(nodes: List[Node], x0: Node = "X0") -> Tuple[List[Node], List[Edge]]:
    if x0 not in nodes:
        nodes = sorted(nodes + [x0])
    return nodes


def bellman_ford(nodes: List[Node], edges: List[Edge], source: Node) -> Dict[Node, float]:
    """
    Input: the distance graph D(S) and the reference node X0.
    Output: shortest path distance from source to each node in the distance graph.
    Raises InfeasibleSTN if a negative cycle is reachable from `source`, but for a consistent STN this should not happen.
    O(||V|| * ||E||) per source node.
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
    """
    Input: distance graph D(S)
    Output: shortest path distances d[u][v] from each node u to each node v in the distance graph.
    complexity is roughly O(|V|^2 * |E|), algorithm might be slow for large graphs.
    """
    d: Dict[Node, Dict[Node, float]] = {}
    for s in nodes:
        d[s] = bellman_ford(nodes, edges, s)
    return d


@dataclass(frozen=True)
class Interval:
    """
    Represents a price interval (left, right] with constant price.
    Use None for -inf (left) or +inf (right).
    """
    left: Optional[float]
    right: Optional[float]
    price: float


def build_price_intervals(landmarks: List[float], prices: List[float]) -> List[Interval]:
    """
    Input: landmarks [l1, l2, ..., lL] and prices [p1, p2, ..., p_{L+1}]
    Output: intervals I1, I2, ..., I_{L+1}

    prices: length L+1, where prices[j] is the price of interval I_{j+1}

    Intervals:
      I1      = (-inf (l0), l1]
      I2      = (l1, l2]
      ...
      I_{L+1} = (l_L, +inf (l_{L+1}))
    """
    if not landmarks:
        if len(prices) != 1:
            raise ValueError("prices must have length 1 when there are no landmarks")
        return [Interval(None, None, prices[0])]
    if sorted(landmarks) != landmarks:
        raise ValueError("landmarks must be sorted increasing")
    if len(prices) != len(landmarks) + 1:
        raise ValueError("prices must have length len(landmarks)+1")

    intervals: List[Interval] = []
    intervals.append(Interval(left=None, right=landmarks[0], price=prices[0]))
    for i in range(1, len(landmarks)):
        intervals.append(Interval(left=landmarks[i - 1], right=landmarks[i], price=prices[i]))
    intervals.append(Interval(left=landmarks[-1], right=None, price=prices[-1]))
    return intervals


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
    end: Node # end node is used for deadline constraints in Alg2, but not directly in Alg1.
    energy: float


@dataclass(frozen=True)
class ActivationTuple:
    """
    Represents the POSET element <Pi, Ij>.
    process Pi starts in interval Ij
    """
    proc_idx: int # index of the process in the input list
    interval_idx: int # index of the interval in the input list
    start_node: Node # Xs_Pi
    left: Optional[float]
    right: Optional[float]
    price: float


def build_activation_tuples(processes: List[Process], intervals: List[Interval]) -> List[ActivationTuple]:
    """
    Input: list of processes and price intervals
    Output: list of all POSET elements <Pi, Ij>
    Create all POSET elements <Pi, Ij> for all processes Pi and all intervals Ij.
    """
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


def poset_precedes(d: Dict[Node, Dict[Node, float]], a: ActivationTuple,  b: ActivationTuple,) -> bool:
    """
    Input: all-pairs shortest paths d(·,·), two activation tuples a=<Pu,Ia> and b=<Pv,Ib>
    Output: boolean of whether activating a would force b to be infeasible.
    binary conflict <Pu, Ia> < <Pv, Ib>  iff  l_b + d(Xs_Pv, Xs_Pu) - l_(a-1) <= 0
    here <Pu, Ia> < <Pv, Ib> means if we have Pv at interval Ib already, then Pu at interval Ia would cause a conflict.

    l_b      = b.right   (right endpoint of interval Ib)
    l_(a-1)  = a.left    (left endpoint of interval Ia)

    If a.left is -inf or b.right is +inf, the expression cannot be <= 0 in the intended sense,
    so we return False (these “infinite endpoint” cases should not create a minimal binary conflict).

    here we are just checking if we picked pv with interval Ib, then whether pu with interval Ia would cause a conflict.
    conflict if the earilest time for process u > the latest time for process v + the time difference between their start nodes.   
    """
    if a.left is None:
        return False
    if b.right is None:
        return False

    latest_b = b.right
    travel = d[b.start_node][a.start_node]
    earliest_a_boundary = a.left
    return latest_b + travel - earliest_a_boundary <= 0.0


def is_non_unary_conflict(d: Dict[Node, Dict[Node, float]], x0: Node, t: ActivationTuple) -> bool:
    """
    Input: all-pairs shortest paths d(·,·), reference node x0, activation tuple t=<Pi, Ij>
    Output: boolean of whether y<Pi,Ij> has a unary conflict
    d(X0, Xs_Pi) - l_(j-1) > 0 and l_j + d(Xs_Pi, X0) >= 0
    d(X0, Xs_Pi) is the earliest time Pi can start, and l_(j-1) is the left endpoint of interval Ij.
    l_j is the right endpoint of interval Ij.
    l_j + d(Xs_Pi, X0) is the latest time Pi can start plus the time difference to get back to X0
    here d(Xs_Pi, X0) is the negative of the earliest time Pi can start
    so we are just doing bidirectional checks
    """
    xs = t.start_node

    # d(X0, xs) - left > 0
    if t.left is None:
        cond1 = True
    else:
        cond1 = (d[x0][xs] - t.left) > 0.0

    # right + d(xs, X0) >= 0
    if t.right is None:
        cond2 = True
    else:
        cond2 = (t.right + d[xs][x0]) >= 0.0

    return cond1 and cond2


def compute_node_weight_cij(processes: List[Process], t: ActivationTuple, max_price: float) -> float:
    """
    Input: list of processes, an activation tuple t=<Pi,Ij>, and the maximum price across all intervals.
    Output: the weight c_ij for the node y<Pi,Ij> in the DAG
    c_ij = (sumW * max_price) - (Wi * price(Ij)) + 1.0
    where Wi is the energy of process Pi, price(Ij) is the price of interval Ij
    and max_price is the maximum price across all intervals.

    Here we are doing a maximum-weight independent set reduction, larger cij means lower original cost.
    Thus here we are subtracting the cost term (Wi * price(Ij)) from a large constant (sumW * max_price) to get the weight c_ij.
    The +1.0 is to ensure that all weights are positive.
    This allows us to solve the MWIS problem with a maxflow algorithm that assumes positive weights.
    """
    sumW = sum(p.energy for p in processes)
    base_weight = sumW * max_price
    Wi = processes[t.proc_idx].energy
    return base_weight - (Wi * t.price) + 1.0


def build_activation_dag(processes: List[Process], tuples_all: List[ActivationTuple], d: Dict[Node, Dict[Node, float]], x0: Node = "X0",
) -> Tuple[List[int], Dict[int, float], List[Tuple[int, int]]]:
    """
    Input: list of processes, list of all activation tuples, all-pairs shortest paths d(·,·), reference node x0
    Output: node set Y (subset of activation tuples), weights c_ij, and edges of G

    here the weights c_ij are for the nodes because we are doing a maximum-weight independent set reduction
    and the edges represent the precedence relation (binary conflicts) between the nodes.
    meaning we only add edges for the tuples with binary conflicts.
    """
    # filter unary conflicts
    Y_ids: List[int] = [idx for idx, t in enumerate(tuples_all) if is_non_unary_conflict(d, x0, t)]

    # compute weights
    max_price = max((t.price for t in tuples_all), default=0.0)
    weight: Dict[int, float] = {}
    for idx in Y_ids:
        weight[idx] = compute_node_weight_cij(processes, tuples_all[idx], max_price)

    # build directed edges
    # This is where we check for binary conflicts
    # Edge construction is O(|Y_ids|^2) because we check all pairs of nodes 
    Y_set = set(Y_ids)
    edges: List[Tuple[int, int]] = []
    for a_idx in Y_ids:
        a = tuples_all[a_idx]
        for b_idx in Y_ids:
            if a_idx == b_idx:
                continue
            b = tuples_all[b_idx]
            # If a < b then add edge y_b -> y_a
            if poset_precedes(d, a, b):
                edges.append((b_idx, a_idx))

    return Y_ids, weight, edges


# MWIS (Max-weight independent set) via bipartite min-cut

class Dinic:
    # we use Dinic because it works on bipartite graph with a time complexity of O(\sqrt(VE)), being the best algorithm so far.
    def __init__(self, n: int):
        # n here refers to the number of nodes in the graph
        self.n = n
        self.adj: List[List[List[float | int]]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: float) -> None:
        # edge weight here is the same as the node weight
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
                    it[u] = i + 1
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
        """
        Input: source node s
        Output: list of booleans indicating which nodes are reachable from s in the residual graph
        Nodes reachable from s in the residual graph (cap > 0).
        """
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


def maximum_weight_independent_set_in_bipartite(Y_ids: List[int], weights: Dict[int, float], dag_edges: List[Tuple[int, int]]) -> Set[int]:
    """
    Input: node set Y (subset of activation tuples), weights c_ij, and edges of G
    Output: Q_G, subset of Y that forms a maximum-weight independent set in G, which is the optimal set of activation tuples.


    Bipartite graph construction:
        We first construct a bipartite graph B = (Y, Y', E') where Y' is a copy of Y.
        For each edge y_u -> y_v in G, we add an edge y_u -> y'_v in B with +inf capacity.
            This means we have y_u already and adding y_v would cause a conflict.
        We also add edges from a source node to each node in Y with capacity equal to the node weight
        and from each node in Y' to a sink node with capacity equal to the node weight.

    Then we compute the minimum cut in B using max_flow method and reachable_from method from the dinic class.
    we know independent set = all nodes - vertex cover.
    so finding min vertex cover = max independent set.
    vertex cover = set of nodes that touches all edges.


    All the reachable nodes from the reachable_from method means they are not selected for the max flow.

    The non-reachable nodes from the reachable_from method means they are in the vertex cover as they are selected for the max flow.
    non-reachable nodes means they have low weight c_ij, thus their original cost is high and we don't want them.

    Then the independent set in G corresponds to the nodes whose left copy is reachable in the residual graph and right copy is not reachable in the residual graph.
    and right copy is not reachable in B, which will be in Q_G.
    Because if it's not reachable on right copy, it means it's not selected for the max flow, thus it has no conflicts with other selected nodes
    and it has a high weight c_ij, thus it's a good choice for the independent set.
    """
    m = len(Y_ids)
    # Compact indexing for left and right copies
    left_index = {tid: i for i, tid in enumerate(Y_ids)}
    right_index = {tid: i for i, tid in enumerate(Y_ids)}

    SRC = 2 * m
    SNK = 2 * m + 1
    # SNK means all the nodes reachable from SNK, which is why it has 2m+1.
    dinic = Dinic(2 * m + 2)
    INF = 1e18

    # Build min-cut network for min-weight vertex cover in bipartite graphs:
    for tid in Y_ids:
        w = weights[tid]
        if w < 0:
            raise ValueError("Algorithm 1 expects positive node weights c_ij; got negative weight.")
        dinic.add_edge(SRC, left_index[tid], w)
        dinic.add_edge(m + right_index[tid], SNK, w)

    # Add bipartite edges based on DAG edges in G
    for u_tid, v_tid in dag_edges:
        if u_tid in left_index and v_tid in right_index:
            dinic.add_edge(left_index[u_tid], m + right_index[v_tid], INF)

    dinic.max_flow(SRC, SNK)
    reach = dinic.reachable_from(SRC)
    # reach gives all the node reachable from SRC in the residual graph.
    # We want the left nodes reachable and the right nodes not reachable.
    # being in the independent set means the activation tuple is the optimal choice for the process
    # because it has no conflicts with other tuples in the independent set and it has a high weight (low cost).

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


# Activate selected tuples

def add_activation_edges_to_distance_graph(edges: List[Edge], x0: Node, t: ActivationTuple) -> List[Edge]:
    """
    Input: current distance graph edges, reference node x0, activation tuple t=<Pi,Ij>
    Output: new distance graph edges with additional edges for activating y<Pi,Ij>
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
    Input: modified distance graph D'(S) with activated tuples, list of processes, reference node x0
    Output: schedule τ* mapping each process start node Xs_Pi to τ*(Xs_Pi)
    Compute shortest path distances d'(Xs_Pi, X0) in D'(S) and output:
    τ*(Xs_Pi) = - d'(Xs_Pi, X0) represent the optimal start time for each process Pi under the activated tuples.
    
    Returns a dict mapping each process start node Xs_Pi to τ*(Xs_Pi).
    """
    schedule: Dict[Node, float] = {}
    for p in processes:
        dist = bellman_ford(nodes, edges, p.start)
        if dist[x0] == float("inf"):
            raise ValueError(f"X0 not reachable from {p.start} in modified graph; cannot compute τ*(Xs_Pi).")
        schedule[p.start] = -dist[x0]
    return schedule


def solve_stn_load_scheduling(stn_constraints: Iterable[STNConstraint], processes: List[Process], landmarks: List[float], prices: List[float], x0: Node = "X0",
) -> Dict[Node, float]:
    """
    Inputs:
      - stn_constraints: STN constraints E
      - processes: list of processes Pi with start/end nodes in the STN and energy Wi
      - landmarks, prices: piecewise-constant price function f(t)

    Output:
      - schedule τ*: dictionary mapping each process start node Xs_Pi to τ*(Xs_Pi)
    """
    nodes, D_edges = build_distance_graph(stn_constraints)
    nodes, D_edges = ensure_reference_node(nodes, D_edges, x0=x0)
    d = all_pairs_shortest_paths(nodes, D_edges)

    intervals = build_price_intervals(landmarks, prices)


    tuples_all = build_activation_tuples(processes, intervals)

    Y_ids, weights, dag_edges = build_activation_dag(processes, tuples_all, d, x0=x0)

    Q_G_lambda_ids = maximum_weight_independent_set_in_bipartite(Y_ids, weights, dag_edges)


    D_prime_edges = list(D_edges)
    for tid in Q_G_lambda_ids:
        D_prime_edges = add_activation_edges_to_distance_graph(D_prime_edges, x0, tuples_all[tid])

    return compute_schedule_from_modified_graph(nodes, D_prime_edges, processes, x0=x0)
