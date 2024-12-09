import numpy as np

class GridWorldEnv:
    def __init__(self):
        # special states (zero-indexed)
        # top left corner is (0, 0)
        self.obstacles = [(2, 2), (3, 2)] 
        self.water_state = (4, 2)
        self.end_state = (4, 4) # terminal state
 
        self.state = (0, 0) # current state
        self.discount = 0.9

        # transition probabilities
        self.success_prob = 0.8
        self.right_prob = 0.05
        self.left_prob = 0.05
        self.stay_prob = 0.1
        self.probs = [self.success_prob, self.right_prob, self.left_prob, self.stay_prob]

    # reset to initial state
    def reset(self):
        self.set_state((0, 0))

    # take a step with the given action
    # returns the next state and reward and whether the agent is now in a terminal state or not
    def step(self, action):
        if self.state == self.end_state: # already in a terminal state
            return (self.state, 0, True)

        outcome = np.random.choice(len(self.probs), p=self.probs) # zero-indexed outcome index
        self.state = self.next_state(outcome, action) # transition to the next state
        
        return (self.state, self.get_reward(self.state), self.state == self.end_state)
    
    # returns the next state coordinates according to the action taken and transition outcome
    def next_state(self, outcome, action):
        r, c = self.state # current row and column
        if outcome == 3: # stay
            return self.state

        next_states = []
        if action == "AU": # in order: success, right, left
            next_states = [(r-1, c), (r, c+1), (r, c-1)] # for each outcome (from transition probabilities)
        elif action == "AD":
            next_states = [(r+1, c), (r, c-1), (r, c+1)]
        elif action == "AL":
            next_states = [(r, c-1), (r-1, c), (r+1, c)]
        elif action == "AR":
            next_states = [(r, c+1), (r+1, c), (r-1, c)]

        state_next = next_states[outcome]
        # clamp within grid limits
        clamped_r = max(min(state_next[0], 4), 0)
        clamped_c = max(min(state_next[1], 4), 0)
        state_next = (clamped_r, clamped_c)
        
        # special interactions with environment
        if state_next in self.obstacles:
            state_next = self.state # don't move
        
        return state_next

    # returns reward for entering this next state
    def get_reward(self, next_state):
        reward = 0
        if next_state == self.end_state:
            reward = 10
        elif next_state == self.water_state:
            reward = -10
        return reward
    
    # ----------------- getters / setters ------------------
    def set_state(self, state):
        self.state = state

    def get_dimensions(self):
        return (5, 5)
    
    def get_discount(self):
        return self.discount
    
    def get_states(self): # no obstacles
        tuples = [(i, j) for i in range(5) for j in range(5)]
        return [tup for tup in tuples if tup not in self.obstacles]
    
    def get_ignored_states(self): # obstacles
        return self.obstacles
    
    def get_ignored_and_end(self):
        return self.obstacles + [self.end_state]
    
    def get_num_actions(self):
        return 4
    
    def index_to_action(self, index):
        return ["AU", "AD", "AL", "AR"][index]
    
    def get_actions(self):
        return ["AU", "AD", "AL", "AR"]
    
    def get_state_from_action_path(self, action_path):
        for action in action_path:
            self.step(action)
        return self.state