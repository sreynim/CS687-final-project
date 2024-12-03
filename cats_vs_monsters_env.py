import numpy as np

class CatsVsMonsters:

    def __init__(self):

        # initialize starting state
        self.state = (0, 0)

        # discount
        self.gamma = 0.925
 
        # MDP probabilities
        self.prob_correct = 0.7
        self.prob_right = 0.12
        self.prob_left = 0.12
        self.prob_remain = 0.06

        # MDP marked states
        self.forbidden_furniture = [(2, 1), (2, 2), (2, 3), (3, 2)] 
        self.monster = [(0, 3), (4, 1)]
        self.food = (4, 4) #terminal state

    
    # returns the next state based on the current state and action taken
    def transition(self, action):

        # already in terminal state
        if self.state == self.food:
            return self.state
        
        # define row and column
        r, c = self.state

        # randomly decide what happens based on probabilities for action being taken
        # order of options - [0: correct, 1: right, 2: left, 3: remain]
        options = [0, 1, 2, 3]
        probabilities = [self.prob_correct, self.prob_right, self.prob_left, self.prob_remain]
        option = np.random.choice(options, p=probabilities)

        # determine all possible next states based on action
        possible_states = [] #[correct, right, left, remain]
        if action == "AU":
            possible_states = [(r-1, c), (r, c+1), (r, c-1), (r, c)]
        elif action == "AD":
            possible_states = [(r+1, c), (r, c-1), (r, c+1), (r, c)]
        elif action == "AL":
            possible_states = [(r, c-1), (r-1, c), (r+1, c), (r, c)]
        elif action == "AR":
            possible_states = [(r, c+1), (r+1, c), (r-1, c), (r, c)]

        # determine next state based on the randomly selected option
        next_state = possible_states[option]

        # if next state is forbidden furniture, cat remains in current state
        if next_state in self.forbidden_furniture:
            return self.state
        
        # if next state is outside the walls, cat remains in current state
        curr_r, curr_c = next_state
        new_r = 0
        new_c = 0
        if (curr_r > 4):
            new_r = 4
            next_state = (new_r, curr_c)
        if (curr_r < 0):
            new_r = 0
            next_state = (new_r, curr_c)
        if (curr_c > 4):
            new_c = 4
            next_state = (curr_r, new_c)
        if (curr_c < 0):
            new_c = 0
            next_state = (curr_r, new_c)
        
        # set current state to next state
        self.state = next_state

        return self.state


    # returns the reward for next state
    def reward(self, next_state):
        if next_state == self.food:
            return 10
        elif next_state == self.monster:
            return -8
        else:
            return -0.05
