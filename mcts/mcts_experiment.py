# experimenting with Monte Carlo Tree Search on 687 Grid World + CatVsMonsters domains

import numpy as np
import sys
import math

from mcts.monte_carlo_tree_search import MCTS
from environments.grid_world_env import GridWorldEnv
from environments.cats_vs_monsters_env import CatsVsMonsters
from environments.dummy_env import DummyEnv
from mcts.action_node import ActionNode

if __name__ == "__main__":
    if sys.argv[1] == "mcts-dummy": # monte carlo tree search on dummy environment (just for testing)
        env = DummyEnv()
        mcts = MCTS(env=env, C=math.sqrt(2), branch_exploration_param=0.5, num_rollouts=20, num_iterations=20)
        root = mcts.run_mcts()
        print(root.get_action())


