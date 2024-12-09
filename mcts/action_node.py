import math

# Node representing an action taken in a grid-world environment
class ActionNode:
    def __init__(self, action, parent):
        self.action = action # string representing action associated with this node
        self.children = [] # child nodes
        self.parent = parent # parent node
        self.playouts = 0 # number of times this ndoe was visited in mcts
        self.total_return = 0 # total returns from playouts of mcts

    # GETTERS

    # return child nodes
    def get_children(self):
        return self.children
    
    # returns this node's parent
    def get_parent(self):
        return self.parent
    
    # returns this node's action
    def get_action(self):
        return self.action
    
    # returns whether this ndoe is a leaf node
    def is_leaf(self):
        return not self.children
    
    # returns the number of playouts (number of times this node was visited in mcts)
    def get_playouts(self):
        return self.playouts

    # returns total return accumulated
    def get_total_return(self):
        return self.total_return

    # MODIFIERS

    # adds given child as a child of this node
    def add_child(self, child):
        self.children.append(child)
    
    # increment playouts amount
    def add_playout(self):
        self.playouts += 1
    
    # add amt to this node's total return amount
    def add_return(self, amt):
        self.total_return += amt
    
    # OTHER

    # returns this node's ucb value
    def get_ucb_val(self, exploration_param):
        return self.get_value() + exploration_param * math.sqrt(math.log(self.parent.get_playouts()) / self.playouts)
    
    # returns this node's value
    def get_value(self):
        if self.playouts <= 0:
            return 0
        else:
            return (self.total_return / self.playouts)

    def __str__(self):
        return f"ActionNode(action={self.action}, parent={self.parent.get_action() if self.parent is not None else "None"}, number of children={len(self.get_children())})"