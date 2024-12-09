import numpy as np
from action_node import ActionNode
import util_mcts

# monte carlo tree search
class MCTS:
    def __init__(self, env, C, branch_exploration_param):
        self.env = env
        self.exploration_param = C
        self.branch_exploration_param = branch_exploration_param
    
    # returns the best (greedy) action from the given root node for num_iterations
    def search_from_root(self, start, num_iterations):
        for i in range(num_iterations):
            print([child.get_action() for child in start.get_children()])
            initial_selected, path_to_node = self.select_node(start)
            selected_node = self.expand(initial_selected) # change selected node to this new child resulting from expansion
            path_to_node_for_action = path_to_node + [selected_node.get_action()] # update path to node to include the new action
            total_return = self.run_simulation(start, selected_node, path_to_node_for_action)
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
            else:
                explore = util_mcts.explore_epsilon_greedy(actions_taken_already, self.env.get_actions(), self.branch_exploration_param) # explore or not
                if explore:
                    not_taken = [a for a in self.env.get_actions() if a not in actions_taken_already]
                    child = ActionNode(np.random.choice(not_taken), node)
                    node = child
                else:
                    node = self.choose_action_ucb(node)
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
    def run_simulation(self, start, node, path_to_node):
        action_path_root_to_start = self.get_action_path_from_initial(start)
        cur_state = self.env.get_state_from_action_path(self.get_action_path_from_initial(start) + path_to_node) # TODO: make it not deterministic

        self.env.set_state(cur_state)

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
        while node is not None: # go "up" the tree
            action_path.append(node.get_action())
            node = node.get_parent()
        return action_path