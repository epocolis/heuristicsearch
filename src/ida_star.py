import random
import heapq
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations


class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds):
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def is_goal(self, state):
        return state == self.goal

    def action_cost(self, s, a, s1):
        return 1

    def h(self, node):
        return 0

    def __str__(self):
        return "{}({!r}, {!r})".format(type(self).__name__, self.initial, self.goal)


class Node:
    "A Node in a search tree."

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(
            state=state, parent=parent, action=action, path_cost=path_cost
        )

    def __repr__(self):
        return "<{}>".format(self.state)

    def __len__(self):
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other):
        return self.path_cost < other.path_cost


failure = Node(
    "failure", path_cost=math.inf
)  # Indicates an algorithm couldn't find a solution.

cutoff = Node(
    "cutoff", path_cost=math.inf
)  # Indicates iterative deepening search was cut off.


def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]


def is_cycle(node, k=30):
    "Does this node form a cycle of length k or less?"

    def find_cycle(ancestor, k):
        return (
            ancestor is not None
            and k > 0
            and (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1))
        )

    return find_cycle(node.parent, k)


class EightPuzzle(Problem):
    """The problem of sliding tiles numbered from 1 to 8 on a 3x3 board,
    where one of the squares is a blank, trying to reach a goal configuration.
    A board state is represented as a tuple of length 9, where the element at index i
    represents the tile number at index i, or 0 if for the empty square, e.g. the goal:
        1 2 3
        4 5 6 ==> (1, 2, 3, 4, 5, 6, 7, 8, 0)
        7 8 _
    """

    def __init__(self, initial, goal=(0, 1, 2, 3, 4, 5, 6, 7, 8)):
        # assert inversions(initial) % 2 == inversions(goal) % 2  # Parity check
        self.initial, self.goal = initial, goal

    def actions(self, state):
        """The indexes of the squares that the blank can move to."""
        moves = (
            (1, 3),
            (0, 2, 4),
            (1, 5),
            (0, 4, 6),
            (1, 3, 5, 7),
            (2, 4, 8),
            (3, 7),
            (4, 6, 8),
            (7, 5),
        )
        blank = state.index(0)
        return moves[blank]

    def result(self, state, action):
        """Swap the blank with the square numbered `action`."""
        s = list(state)
        blank = state.index(0)
        s[action], s[blank] = s[blank], s[action]
        return tuple(s)

    def h1(self, node):
        """The misplaced tiles heuristic."""
        return hamming_distance(node.state, self.goal)

    def h2(self, node):
        """The Manhattan heuristic."""
        X = (0, 1, 2, 0, 1, 2, 0, 1, 2)
        Y = (0, 0, 0, 1, 1, 1, 2, 2, 2)
        return sum(
            abs(X[s] - X[g]) + abs(Y[s] - Y[g])
            for (s, g) in zip(node.state, self.goal)
            if s != 0
        )

    def h(self, node):
        return self.h2(node)


def hamming_distance(A, B):
    "Number of positions where vectors A and B are different."
    return sum(a != b for a, b in zip(A, B))


def inversions(board):
    "The number of times a piece is a smaller number than a following piece."
    return sum((a > b and a != 0 and b != 0) for (a, b) in combinations(board, 2))


def board8(board, fmt=(3 * "{} {} {}\n")):
    "A string representing an 8-puzzle board"
    return fmt.format(*board).replace("0", "_")


class Board(defaultdict):
    empty = "."
    off = "#"

    def __init__(self, board=None, width=8, height=8, to_move=None, **kwds):
        if board is not None:
            self.update(board)
            self.width, self.height = (board.width, board.height)
        else:
            self.width, self.height = (width, height)
        self.to_move = to_move

    def __missing__(self, key):
        x, y = key
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return self.off
        else:
            return self.empty

    def __repr__(self):
        def row(y):
            return " ".join(self[x, y] for x in range(self.width))

        return "\n".join(row(y) for y in range(self.height))

    def __hash__(self):
        return hash(tuple(sorted(self.items()))) + hash(self.to_move)


failure = Node(
    "failure", path_cost=math.inf
)  # Indicates an algorithm couldn't find a solution.

cutoff = Node("cutoff", path_cost=math.inf)  # Ind


def g(node):
    return node.path_cost


def iterative_deepening_a_star(problem, weight):
    """
    Performs the iterative deepening A Star (A*) algorithm to find the shortest path from a start to a target node.
    Can be modified to handle graphs by keeping track of already visited nodes.
    :param problem:      An adjacency-matrix-representation of the tree where (x,y) is the weight of the edge or 0 if there is no edge.
    :param wieght:      The weight to be applied to the heuristics.
    :return: the path to the goal.
    """
    start = Node(problem.initial)
    threshold = g(start) + weight * problem.h(start)
    while True:
        result_node = iterative_deepening_a_star_rec(problem, start, threshold)
        if result_node == failure:
            # Node not found and no more nodes to visit
            return failure
        elif problem.is_goal(result_node.state):
            # if we found the node, the function returns the negative distance
            print("Found the node we're looking for!")
            return result_node
        else:
            # if it hasn't found the node, it returns the (positive) next-bigger threshold

            threshold = threshold + 1


static_count = 1


def iterative_deepening_a_star_rec(problem, node, threshold):
    """
    Performs DFS up to a depth where a threshold is reached (as opposed to interative-deepening DFS which stops at a fixed depth).
    Can be modified to handle graphs by keeping track of already visited nodes.
    :param problem: A sliding puzzle problem.
    :param node:    The node to continue from.
    :param threshold: The current f_cost
    :return: the path to the goal.
    """

    global static_count
    static_count = static_count + 1

    if problem.is_goal(node.state):
        # We have found the goal node we we're searching for
        print(f"found goal:{node}")
        sys.exit(0)
        return node

    estimate = g(node) + problem.h(node)
    if estimate > threshold:
        return cutoff

    min = failure
    for child_node in expand(problem, node):
        if problem.is_goal(child_node.state):
            # Node found
            print(path_states(child_node))
            sys.exit(0)
            return child_node

        t = iterative_deepening_a_star_rec(problem, child_node, threshold)
        if problem.is_goal(child_node.state):
            # Node found
            return child_node
        elif t == cutoff:
            min = t

    return min


def node_expansion_weight_relationship(algorithm_func, problem_list, weight_list):
    for problem in problem_list:
        for weight in weight_list:
            print(algorithm_func(problem, weight))
            break
        break


def weight_total_cost_relationship(algorithm_func, problem_list, weight_list):
    for problem in problem_list:
        for weight in weight_list:
            algorithm_func(problem, weight)


def solve_problem(algorithm_func, problem, weight):
    algorithm_func(problem, weight)


if __name__ == "__main__":
    # Some specific EightPuzzle problems
    problem_list = [
        EightPuzzle((1, 4, 2, 0, 7, 5, 3, 6, 8)),
        EightPuzzle((1, 2, 3, 4, 5, 6, 7, 8, 0)),
        EightPuzzle((4, 0, 2, 5, 1, 3, 7, 8, 6)),
        EightPuzzle((7, 2, 4, 5, 0, 6, 8, 3, 1)),
        EightPuzzle((8, 6, 7, 2, 5, 4, 3, 0, 1)),
        EightPuzzle((4, 0, 2, 8, 6, 5, 3, 1, 7)),
        EightPuzzle((1, 2, 3, 8, 5, 4, 6, 0, 7)),
        EightPuzzle((0, 6, 3, 5, 2, 8, 1, 4, 7)),
        EightPuzzle((3, 7, 5, 6, 4, 0, 2, 1, 8)),
        EightPuzzle((0, 8, 2, 7, 4, 5, 3, 6, 1)),
        EightPuzzle((1, 4, 5, 8, 3, 7, 0, 2, 6)),
        EightPuzzle((5, 6, 4, 0, 8, 2, 7, 3, 1)),
        EightPuzzle((5, 3, 8, 1, 0, 4, 6, 7, 2)),
        EightPuzzle((5, 0, 4, 6, 3, 1, 7, 8, 2)),
    ]

    # how do you convert this to IDA*
    # weight_list = [1,1.5,2,5,15,35]
    # node_expansion_weight_relationship(iterative_deepening_a_star, problem_list, weight_list)

    # node_expansion_weight_relationship(anytime_iterative_deepening_a_star, problem_list, weight_list)

    # weight_total_cost_relationship(iterative_deepening_a_star, problem_list, weight_list)

    # weight_total_cost_relationship(anytime_iterative_deepening_a_star, problem_list, weight_list)

    # total cost vs weight for each algorithm
    # i.e the number of paths in the solution

    """
    Run an algorithm on a problem
    """
    weight = 15
    problem = problem_list[13]
    algorithm_func = iterative_deepening_a_star

    goal_test = number_of_expansions
    total_cost = the_number_of_moves_in the solution

    solve_problem(algorithm_func, problem, weight)
