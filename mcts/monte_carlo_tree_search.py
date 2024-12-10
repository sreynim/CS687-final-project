import numpy as np
from mcts.action_node import ActionNode
import mcts.util_mcts as util_mcts

# monte carlo tree search
class MCTS:
    def __init__(self, env, C, branch_exploration_param, num_rollouts, num_iterations, initial_state, epsilon):
        self.env = env
        if not self.env.in_bounds(initial_state):
            raise ValueError("initial_state is not in environment bounds")
        self.exploration_param = C
        self.branch_exploration_param = branch_exploration_param
        self.num_rollouts = num_rollouts
        self.num_iterations = num_iterations
        self.initial_state = initial_state
        self.epsilon = epsilon
    
    # run monte carlo tree search for num_iterations (different starting nodes), each iteration having num_rollouts from the same node
    # returns the root node of the entire tree
    def run_mcts(self, init_state):
        root_node = ActionNode(None, None)
        starting_node = root_node
        for i in range(self.num_iterations):
            print("iteration: " + str(i + 1))
            starting_node = self.search_from_root(starting_node, init_state)
        return root_node
    
    # returns the best (greedy) action from the given root node for num_iterations
    def search_from_root(self, start, init_state):
        for i in range(self.num_rollouts):
            # print("--- rollout: " + str(i + 1))
            initial_selected, path_to_node = self.select_node(start)
            selected_node = self.expand(initial_selected) # change selected node to this new child resulting from expansion
            path_to_node_for_action = path_to_node + [selected_node.get_action()] # update path to node to include the new action
            total_return = self.run_simulation(start, path_to_node_for_action, init_state)
            self.backup(selected_node, total_return)
        return self.choose_action_ucb(start)
            
    # select a leaf node by traversing from the given node, using the UCB criterion
    # might add new nodes for actions that have not been taken, using branch_exploration_param (sort of like epsilon-soft policy, but for deciding whether to add a random new action or not)
    def select_node(self, node):
        action_path = []
        while not node.is_leaf():
            action_path.append(node.get_action())
            actions_taken_already = [child.get_action() for child in node.get_children()]
            if len(actions_taken_already) == len(self.env.get_actions()):
                node = self.choose_action_ucb(node) # from existing nodes
            else: # we havent explored all actions yet
                explore = util_mcts.explore_epsilon_greedy(actions_taken_already, self.env.get_actions(), self.branch_exploration_param) # choose unexplored or one already explored?
                if explore:
                    not_taken = [a for a in self.env.get_actions() if a not in actions_taken_already]
                    child = ActionNode(np.random.choice(not_taken), node)
                    node.add_child(child)
                    node = child
                else: # from existing nodes
                    node = self.choose_action_ucb(node)
        action_path.append(node.get_action())
        return (node, action_path)
    
    # choose an action / child node using UCB criterion (upper confidence bound)
    def choose_action_ucb(self, node):
        children = node.get_children()
        if len(children) == 0:
            return node
        
        ucb_vals = np.array([child.get_ucb_val(self.exploration_param) for child in children])
        # if node.get_action() is None:
        #     print("jkcnkdjnc")
        #     print([child.get_action() for child in children])
        #     print(ucb_vals)
        return children[util_mcts.argmax(ucb_vals)]
    
    # add random child node / action to the given node (the given node is guaranteed to be a leaf node)
    def expand(self, node):
        action = np.random.choice(self.env.get_actions())
        child = ActionNode(action, node)
        node.add_child(child)
        return child

    # returns the total discounted return in env for one full simulation / episode from the given node
    # using random actions
    def run_simulation(self, start, path_to_node, init_state):
        self.env.set_state(init_state)
        action_path_root_to_start = self.get_action_path_from_initial(start)
        full_path = action_path_root_to_start + path_to_node
        full_path = [a for a in full_path if a is not None]
        cur_state = self.env.get_state_from_action_path(full_path)

        total_discounted_rewards = 0
        terminated = False
        iteration = 0
        while not terminated:
            action = np.random.choice(self.env.get_actions())
            next_state, reward, is_terminal = self.env.step(action)
            total_discounted_rewards += (self.env.get_discount() ** iteration) * reward 
            iteration += 1
            cur_state = next_state
            if is_terminal:
                terminated = True
        return total_discounted_rewards

    # update / initialize the values of nodes on the path to node using the return amount from a simulated episode
    def backup(self, node, return_amt):
        while node is not None:
            node.add_playout()
            node.add_return(return_amt)
            node = node.get_parent()
    
    # returns the action path to get from initial state of env to this node
    def get_action_path_from_initial(self, node):
        node = node.get_parent()
        action_path = []
        while node is not None: # go "up" the    tree
            action_path.append(node.get_action())
            node = node.get_parent()
        return action_path

    # returns the total discounted return from following the epsilon-soft policy from the learned mcts tree's grid (representing something like "q-values")
    def evaluate_mcts_epsilon_soft_policy(self, grid):
        # run simulation using found policy
        self.env.set_state(self.env.get_init_state())
        cur_state = self.env.get_cur_state()

        total_discounted_rewards = 0
        terminated = False
        iteration = 0
        while not terminated:
            grid_state_info = grid[cur_state[0]][cur_state[1]]
            possible_actions = grid_state_info[1]
            possible_actions_vals = grid_state_info[0]
            action = util_mcts.epsilon_greedy_action(possible_actions, possible_actions_vals, self.epsilon)
            next_state, reward, is_terminal = self.env.step(action)
            total_discounted_rewards += (self.env.get_discount() ** iteration) * reward 
            iteration += 1
            cur_state = next_state
            if is_terminal:
                terminated = True
        return total_discounted_rewards
        
    # returns an array containing a tuple (list for each state of "state-values" of each of its children, and list of corresponding actions) for each state in env
    def get_child_vals_for_each_state(self):
        dim_r, dim_c = self.env.get_dimensions()
        grid = [[] for _ in range(dim_r)]
        for r in range(dim_r):
            for c in range(dim_c):
                root = self.run_mcts((r, c))
                actions = [child.get_action() for child in root.get_children()]
                vals = [child.get_value() for child in root.get_children()]
                grid[r].append((vals, actions))
        return grid

    # returns the best (greedy) action from all the nodes using "q-values" from grid (deterministic policy)
    def get_greedy_policy_from_nodes(self, grid):
        dim_r, dim_c = self.env.get_dimensions()
        policy = [[] for _ in range(dim_r)]
        for r in range(dim_r):
            for c in range(dim_c):
                actions = grid[r][c][1]
                vals = grid[r][c][0]
                argmax = util_mcts.argmax(vals)
                policy[r].append(actions[argmax])
        return policy