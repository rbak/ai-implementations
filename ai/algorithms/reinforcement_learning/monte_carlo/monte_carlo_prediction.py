import numpy as np
from collections import defaultdict
from ai.algorithms.base import BaseAlgorithm

class MonteCarloPrediction(BaseAlgorithm):
    def __init__(self, env, action=True, first_visit=True):
        """Monte Carlo Prediction initialization

        Monte Carlo prediction learning.  Policies are not re-evaluated during the run.
        This supports learning both state and action values, as well as both first-visit
        and every-visit Monte Carlo.

        Arguments:
            env -- An environment to run the algorithm against.

        Keyword Arguments:
            action {bool} -- If true, search for the action value function.
                             If false, search for the state value function (default: True)
            first_visit {bool} -- If true, only evaluate the first visit to a state.
                                  If false, average all visits to a state. (default: True)
        """
        self.env = env
        self.action = action
        self.first_visit = first_visit

    def run(self, num_episodes=50000, gamma=1.0):
        """Runs the Monte Carlo algorithm

        Runs Monte Carlo against the initialized evironment.

        Keyword Arguments:
            num_episodes {int} -- The number of episodes to run for. (default: 50000)
            gamma {float} -- The discount rate. (default: 1.0)
        """
        # initialize
        N = defaultdict(lambda: 0)
        return_sums = defaultdict(lambda: 0)
        # loop over episodes
        for i_episode in range(1, num_episodes + 1):
            episode = []
            state = self.env.reset()
            # generate an episode
            while True:
                action = np.random.choice(np.arange(2))
                next_state, reward, done, info = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state
                if done:
                    break
            # parse episode
            states, actions, rewards = zip(*episode)
            discounts = np.array([gamma**i for i in range(len(rewards) + 1)])
            visited = []
            if self.action:
                steps = zip(states,actions)
            else:
                steps = states
            for i, step in enumerate(steps):
                if self.first_visit and step in visited:
                    continue
                N[step] += 1
                return_sums[step] += sum(rewards[i:] * discounts[:-(1 + i)])

        # If self.action, state here is a state-action pair
        value_function = {state: (return_sums[state]/N[state]) for state in N}
        self.env.visualize_value(value_function)
