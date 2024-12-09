import numpy as np
# returns probabilities for epsilon greedy policy for mcts nodes
def get_episilon_greedy_probs(actions_taken_already, actions, epsilon):
    num_actions = len(actions)
    num_taken = len(actions_taken_already)
    prob_func = lambda x: ((1 - epsilon) / num_taken) + (epsilon / num_actions) if x in actions_taken_already else (epsilon / num_actions)
    vectorized = np.vectorize(prob_func) # val to prob

    return vectorized(actions)

# returns whether to explore new actions or not (for mcts)
def explore_epsilon_greedy(actions_taken_already, actions, epsilon):
    probs = get_episilon_greedy_probs(actions_taken_already, actions, epsilon)
    choice = np.random.choice(a=actions, p=probs)
    return choice not in actions_taken_already