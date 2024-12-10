# experimenting with Monte Carlo Tree Search on 687 Grid World + CatVsMonsters domains

import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import time

from mcts.monte_carlo_tree_search import MCTS
from environments.grid_world_env import GridWorldEnv
from environments.cats_vs_monsters_env import CatsVsMonsters
from environments.dummy_env import DummyEnv
from mcts.action_node import ActionNode
import mcts.util_mcts as util_mcts

# perform mcts experiments on the given environment
def experiment(env):
    default_iterations = 500
    default_rollouts = 500
    default_C = math.sqrt(2)
    default_epsilon = 0.1
    default_branch_param = 0.5
    init_state = env.get_init_state()
    # experiment with different branch exploration parameters
    branch_param_arr = [0.3, 0.5, 0.7]
    branch_returns = []
    for branch_param in branch_param_arr:
        mcts = MCTS(env=env, C=default_C, branch_exploration_param=branch_param, num_rollouts=default_rollouts, num_iterations=default_iterations, initial_state=init_state, epsilon=default_epsilon)
        grid = mcts.get_child_vals_for_each_state()
        branch_returns.append(mcts.evaluate_mcts_epsilon_soft_policy(grid))
    print("branch returns = " + str(branch_returns))
    # experiment with different UCB exploration params ("C")
    C_arr = [0.5, 1, 2, 5]
    C_returns = []
    for C_val in C_arr:
        mcts = MCTS(env=env, C=C_val, branch_exploration_param=default_branch_param, num_rollouts=default_rollouts, num_iterations=default_iterations, initial_state=init_state, epsilon=default_epsilon)
        grid = mcts.get_child_vals_for_each_state()
        C_returns.append(mcts.evaluate_mcts_epsilon_soft_policy(grid))
    print("C returns = " + str(C_returns))
    # experiment with different epsilon values
    epsilon_arr = [0.1, 0.3, 0.5, 0.7, 0.9]
    epsilon_returns = []
    for ep in epsilon_arr:
        mcts = MCTS(env=env, C=default_C, branch_exploration_param=default_branch_param, num_rollouts=default_rollouts, num_iterations=default_iterations, initial_state=init_state, epsilon=ep)
        grid = mcts.get_child_vals_for_each_state()
        epsilon_returns.append(mcts.evaluate_mcts_epsilon_soft_policy(grid))
    print("epsilon returns = " + str(epsilon_returns))
    # experiment with different num_iterations / num_rollouts numbers
    best_num = get_return_vs_iterations_graph(env=env, C=default_C, branch_exploration_param=default_branch_param, num_rollouts=num, num_iterations=num, initial_state=init_state, epsilon=default_epsilon)

    # now identify the best parameters to get a final graph
    best_branch_param = branch_param_arr[util_mcts.argmax(branch_returns)]
    best_C = C_arr[util_mcts.argmax(C_returns)]
    best_epsilon = epsilon_arr[util_mcts.argmax(epsilon_returns)]
    print("best_branch_param = " + str(best_branch_param))
    print("best_C = " + str(best_C))
    print("best_epsilon = " + str(best_epsilon))
    print("best number of rollouts and iterations = " + str(best_num))
    best_num = get_return_vs_iterations_graph(env=env, C=best_C, branch_exploration_param=best_branch_param, initial_state=init_state, epsilon=best_epsilon)
    print("final best number for rollouts and iterations is " + str(best_num))

    # get final greedy policy
    mcts = MCTS(env=env, C=best_C, branch_exploration_param=best_branch_param, num_rollouts=best_num, num_iterations=best_num, initial_state=init_state, epsilon=best_epsilon)
    grid = mcts.get_child_vals_for_each_state()
    policy = mcts.get_greedy_policy_from_nodes(grid)
    util_mcts.print_arr(policy)

# experiment with different rollout / iterations vals and plot vs return from policy learned from tree
def get_return_vs_iterations_graph(env, C, branch_exploration_param, initial_state, epsilon):
    num_arr = [i * 100 for i in range(1, 21)]
    sum_returns = np.zeros((1, 20))
    sum_times = np.zeros((1, 20))
    for i in range(50):
        num_returns = []
        eval_times = []
        for num in num_arr:
            mcts = MCTS(env=env, C=C, branch_exploration_param=branch_exploration_param, num_rollouts=num, num_iterations=num, initial_state=initial_state, epsilon=epsilon)
            grid = mcts.get_child_vals_for_each_state()
            start_time = time.time()
            num_returns.append(mcts.evaluate_mcts_epsilon_soft_policy(grid))
            end_time = time.time()
            eval_times.append(end_time - start_time)
        sum_returns = sum_returns + np.array(num_returns)
        sum_times = sum_times + np.array(eval_times)
    avg_returns = sum_returns / 50
    avg_times = sum_times / 50
    print("avg_returns = " + str(avg_returns))
    plot(num_arr, avg_returns, "Num Iterations / Rollouts", "Total discounted return from policy", "Return for Number of Iterations / Rollouts")
    plot(num_arr, avg_times, "Num Iterations / Rollouts", "Time taken to evaluate policy (s)", "Time needed to evaluate policy from num iterations / rollouts")
    return num_arr[util_mcts.argmax(avg_returns)]

def plot(x, y, x_lab, y_lab, title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.show()

if __name__ == "__main__":
    if sys.argv[1] == "mcts-dummy": # monte carlo tree search on dummy environment (just for testing)
        env = DummyEnv()
        experiment(env)


