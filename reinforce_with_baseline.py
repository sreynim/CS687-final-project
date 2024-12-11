import numpy as np
import util_rwb as util_rwb


# reinforce with baseline
class RWB:
    def __init__(self, env, initial_state, alpha, b):
        self.env = env
        if not self.env.in_bounds(initial_state):
            raise ValueError("initial_state is not in environment bounds")
        self.initial_state = initial_state
        self.alpha = alpha     
        self.b = b             
        # initialize policy parameterization
        self.policy_params = {}
        for state in env.get_states():
            self.policy_params[state] = np.zeros(len(env.get_actions()))
        # initialize baseline parameterization
        self.baseline_params = {}
        for state in env.get_states():
            self.baseline_params[state] = 0


    # run the episodes for reinforce with baseline
    def run_rwb(self, episodes):
        for ep in range(episodes):
            # generate the episode
            episode = self.create_episode()
            # make the updates to the parameters
            self.make_updates_in_episode(episode)
            #print(episode)


    # create an episode and return it, contains states, actions, and rewards
    def create_episode(self):
        episode = []
        # initialize state
        state = self.env.get_init_state()
        continue_next = True
        while continue_next:
            # select the action
            chosen_action = self.select_action(state)
            action = self.env.index_to_action(chosen_action)
            next_state, reward, terminal = self.env.step(action)
            # add next state, action, and reward to episode
            episode.append((state, chosen_action, reward))
            #if terminal state, end loop
            if terminal:
                continue_next = False
            else:
                state = next_state
        return episode


    # select an action randomly and return it
    def select_action(self, state):
        # use softmax to get probabilities for each action
        action_probabilities = util_rwb.softmax((self.policy_params[state]))
        # randomly select action
        chosen_action = np.random.choice(len(action_probabilities), p=action_probabilities)
        return chosen_action
    

    # make the updates for the step sizes in the policy and baseline parameters
    def make_updates_in_episode(self, episode):
        G = 0   # return
        for curr in reversed(range(len(episode))):
            state, chosen_action, reward = episode[curr]
            # compute return
            G = G * self.env.get_discount() + reward
            #print(G)
            #compute gradient
            # update policy parameter
            self.policy_params[state] += self.alpha # * gradient
            #print(self.policy_params[state])
            # update baseline parameter
            baseline = self.baseline_params[state]
            self.baseline_params[state] += self.b * (G - baseline)
            #print(self.baseline_params[state])