import numpy as np
# returns probabilities for epsilon greedy policy for mcts nodes, but specifically for exploring different branches
def get_episilon_greedy_probs_exploration(actions_taken_already, actions, epsilon):
    num_actions = len(actions)
    num_taken = len(actions_taken_already)
    prob_func = lambda x: ((1 - epsilon) / num_taken) + (epsilon / num_actions) if x in actions_taken_already else (epsilon / num_actions)
    vectorized = np.vectorize(prob_func) # val to prob

    return vectorized(actions)

# returns whether to explore new actions or not (for mcts), specifically for exploring different branches
def explore_epsilon_greedy(actions_taken_already, actions, epsilon):
    probs = get_episilon_greedy_probs_exploration(actions_taken_already, actions, epsilon)
    choice = np.random.choice(a=actions, p=probs)
    return choice not in actions_taken_already

# returns probabilities for epsilon greedy policy
def get_episilon_greedy_probs(actions, vals, epsilon):
    num_actions = len(actions)
    best_val = np.max(vals)
    num_best = len(np.where(vals == best_val)[0])

    prob_func = lambda x: ((1 - epsilon) / num_best) + (epsilon / num_actions) if x == best_val else (epsilon / num_actions)
    vectorized = np.vectorize(prob_func) # val to prob

    return vectorized(vals)

# returns an action index based on epsilon greedy policy
def epsilon_greedy_action(actions, vals, epsilon):
    probs = get_episilon_greedy_probs(actions, vals, epsilon)
    return np.random.choice(a=actions, p=probs)

# returns the argmax (index) of the given array
# if there are multiple, returns one of the best indices uniformly at random
def argmax(arr):
    best_val = np.max(arr)
    best_indices = np.argwhere(arr == best_val)[0]
    return np.random.choice(best_indices)

# prints a tree
def print_tree(node, level=0):
    print(" " * level * 2 + str(node.get_action()))
    # recursion to print child nodes
    for child in node.get_children():
        print_tree(child, level + 1)
    
# prints array nicely
def print_arr(arr):
    for row in arr:
        print(row)