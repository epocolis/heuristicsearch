---

marp: true
pagination: true

---

# Course: Heauristic Search 
## An empirical analysis of the run time complexity of the Anytime WIDA$^*$ and the $WIDA^*$ search algorithms. 

Presented by: Leotis Buchanan
Date: December 5,2022

---

# Project Questions

1. What is the impact on the run time complexity of the WIDA* by converting it to $Anytime \ WIDA^*$ 

2. What are performance differences between the two algorithms? 

---

# Introduction 






---
# Methodology 

1. Implement the WIDA$^*$ algorithm 
   - For this I reused the code from [2] 

2. Implement the Anytime WIDA$^*$ algorithm. The approach presented in [1] was used to transform the WIDA$^*$ to Anytime WIDA$^*$ 

---

# Method for converting an Heuristic Search algorithm to and Anytime algorithm

The approach consists of 3 changes as stated in [1]: 

1) A non-admissible evaluation function, $f'(n) = g(n) + h'(n)$,where $h'(n)$ is not admissable, is used to select nodes for expansion in an order that allows good, but possibly suboptimal, solutions to be found quickly. 

2) The search continues after a solution is found, in order to find improved solutions.

3) An admissible evaluation function $f(n) = g(n) + h(n)$, where $h(n)$ is admissible,
 is used together with an upper bound on the optimal solution cost.
 This cost is given by the best solution found so far. This is done in order to prune the search space and detect convergence to an optimal solution. 

---
# WIDA* Algorithm 

WIDA$^*$ works as follows: 
- At each iteration, perform a depth-first search, cutting off a branch when its total cost $f(n) = g(n) + h(n)$ exceeds a given threshold.
- This threshold starts at the estimate of the cost at the initial state, and increases for each iteration of the algorithm. 
- At each iteration the threshold used for the next iteration is the minimum cost of all values that exceeded the current threshold.[2]

---

# The WIDA* Algorithm Implementation

```python

def iterative_deepening_a_star(problem, weight):
    start_node = Node(problem.initial)
    f_cost_bound = g(start_node) + weight * problem.h(start_node)
    while True:
        result_node = search(problem, start, f_cost_bound)
        if result_node == failure:
            # Node not found and no more nodes to visit
            return failure
        elif problem.is_goal(result_node.state):
            return result_node
        else:
           f_cost_bound = f_cost_bound + 1
```
---

```python
def search(problem, node,f_cost_bound ):
    
    if problem.is_goal(node.state):
        # We have found the goal node we we're searching for
        return node

    estimate = g(node) + problem.h(node)
    if estimate > f_cost_bound:
        return cutoff

    min = failure
    for child_node in expand(problem, node):
        t = iterative_deepening_a_star_rec(problem, child_node, f_cost_bound)
        if problem.is_goal(child_node.state):
            # Node found
            print(path_states(child_node))
            return child_node
        elif t == cutoff:
            min = t

    return min

``` 
--- 

# Anytime WIDA$^*$ Algorithm Implementation

```python
def any_time_wida_star(problem): 
  g = 0 

  start_node = Node(problem.initial)

  f_start = problem.h(start)

  f_w_start = w * problem.h(start)

  incumbent_solution_node = None

  while(true) and not interrupted do:  raise NotImplementedError

  return error_bound, incumbent_solution_node

```  

--- 

# Experiment Setup 

The following 8 sliding tile puzzles problems will be used to evaluate the Anytime WIDA$^*$ algorithm.  

![](https://ece.uwaterloo.ca/~dwharder/aads/Algorithms/N_puzzles/images/puz3.png)

---

# The python implementation are shown below

```python

    e1 = EightPuzzle((1, 4, 2, 0, 7, 5, 3, 6, 8))
    e2 = EightPuzzle((1, 2, 3, 4, 5, 6, 7, 8, 0))
    e3 = EightPuzzle((4, 0, 2, 5, 1, 3, 7, 8, 6))
    e4 = EightPuzzle((7, 2, 4, 5, 0, 6, 8, 3, 1))
    e5 = EightPuzzle((8, 6, 7, 2, 5, 4, 3, 0, 1))
    e6 = EightPuzzle((1, 2, 0, 8, 5, 3, 6, 7, 4))
    e7 = EightPuzzle((4, 0, 2, 8, 6, 5, 3, 1, 7))
    e8 = EightPuzzle((1, 2, 3, 8, 5, 4, 6, 0, 7))
    e9 = EightPuzzle((0, 6, 3, 5, 2, 8, 1, 4, 7))
    e10 = EightPuzzle((3, 7, 5, 6, 4, 0, 2, 1, 8))
    e11 = EightPuzzle((0, 8, 2, 7, 4, 5, 3, 6, 1))
    e12 = EightPuzzle((1, 4, 5, 8, 3, 7, 0, 2, 6))
    e13 = EightPuzzle((5, 6, 4, 0, 8, 2, 7, 3, 1))
    e14 = EightPuzzle((5, 3, 8, 1, 0, 4, 6, 7, 2))
    e15 = EightPuzzle((5, 0, 4, 6, 3, 1, 7, 8, 2))

```    

--- 

# Performance and evaluation

See the assigments 

what plots should I create 

- Plot the charts for the WIDA* 

1. For WIDA* and Anytime WIDA* 

Create a line chart which shows the total number of node expansions over all
15 problems when using each algorithm and each weight. There should be one line for
each algorithm, and the chart should show the relationship between weight and total
number of nodes expanded


- if Anytime WIDA* is not implemented leave space for it and mention that it will be  done and included in the report. 



---
# Conclusion and future work

Answer the following questions 

1. What is the impact on the run time complexity of the WIDA* by converting it to $Anytime \ WIDA^*$ 

2. What are performance differences between the two algorithms? 


Briefly discuss the issues 
























---

# References

[1]  HANZEN & ZHOU 
[2]  AIMA [right full reference]






# Results 





# Further work 




