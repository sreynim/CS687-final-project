# experimenting with Monte Carlo Tree Search on 687 Grid World + CatVsMonsters domains

import numpy as np
import sys
import math

from mcts.monte_carlo_tree_search import MCTS
from environments.grid_world_env import GridWorldEnv
from environments.cats_vs_monsters_env import CatsVsMonsters
from environments.dummy_env import DummyEnv
from mcts.action_node import ActionNode
import mcts.util_mcts as util_mcts

if __name__ == "__main__":
    if sys.argv[1] == "mcts-dummy": # monte carlo tree search on dummy environment (just for testing)
        env = DummyEnv()
        mcts = MCTS(env=env, C=math.sqrt(2), branch_exploration_param=0.3, num_rollouts=200, num_iterations=200, initial_state=(0, 0), epsilon=0.2)
        grid = mcts.get_child_vals_for_each_state()
        policy = mcts.get_greedy_policy_from_nodes(grid)
        util_mcts.print_arr(policy)


