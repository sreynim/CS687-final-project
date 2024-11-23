import numpy as np

# environment for the inverted pendulum domain
class PendulumEnv:
    G = 10 # gravity
    L = 1 # length of pendulum
    M = 1 # mass of pendulum
    MAX_SPEED = 8 # maximum angular velocity
    MAX_TORQUE = 2 # maximum torque
    dt = 0.05 # time step
    INITIAL_STATE_UPPER_BOUND = (5 / 6) * np.pi # upper bound of interval to sample initial state from

    def __init__(self):
        self.reset_state()
    
    # reset the state
    def reset_state(self):
        self.cur_angle = np.random.uniform(-self.INITIAL_STATE_UPPER_BOUND, self.INITIAL_STATE_UPPER_BOUND)
        self.cur_velocity = np.random.uniform(-1, 1)
    
    # transition to the next state given the action (torque applied to the pivot)
    def step(self, action):
        next_acceleration = (3 * self.G / 2 * self.L) * np.sin(self.cur_angle) + (3 * action / self.M * (self.L ** 2))
        next_velocity = np.clip(self.cur_velocity + next_acceleration * self.dt, -self.MAX_SPEED, self.MAX_SPEED)
        next_angle = self.cur_angle + next_velocity * self.dt

        self.cur_angle = next_angle
        self.cur_velocity = next_velocity
    
    # returns the return from the episode given the neural network nn
    def get_return(self, nn):
        self.reset_state()
        total_discounted_rewards = 0
        for t in range(200):
            action = nn.get_action(np.array([self.cur_angle, self.cur_velocity], dtype=np.float64))
            reward = self.get_reward(action)
            total_discounted_rewards += reward
            self.transition_to_next_state(action)
        return total_discounted_rewards
    
    # returns the reward from performing this action (torque applied to the pivot)
    def get_reward(self, action):
        w_norm = ((self.cur_angle + np.pi) % (2 * np.pi)) - np.pi
        return -((w_norm ** 2) + 0.1 * (self.cur_velocity ** 2) + 0.001 * (action ** 2)) # the reward