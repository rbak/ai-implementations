import numpy as np
from collections import defaultdict
from ai.algorithms.base import BaseAlgorithm

class MonteCarloControl(BaseAlgorithm):
    """ Monte Carlo Control learning.

        Supports both first-visit and every-visit Monte Carlo."""
    hyper_parameters = ['alpha', 'gamma']

    def __init__(self, env, constant_alpha=True, first_visit=True):
        """Monte Carlo Control initialization

        Arguments:
            env -- An environment to run the algorithm against.

        Keyword Arguments:
            constant_alpha {bool} -- If true, updates to Q use a constant factor rather than
                decreasing according to how many times a state is visited. (default: True)
            first_visit {bool} -- If true, only evaluate the first visit to a state.
                If false, average all visits to a state. (default: True)
        """
        super().__init__()
        self.env = env
        self.constant_alpha = constant_alpha
        self.first_visit = first_visit

    def run(self, num_episodes=50000, **kwargs):
        super().run(**kwargs)
        """Runs the Monte Carlo algorithm

        Runs Monte Carlo against the initialized evironment.

        Keyword Arguments:
            num_episodes {int} -- The number of episodes to run for. (default: 50000)
        """
        # initialize
        N = defaultdict(lambda: 0)
        Q = defaultdict(lambda: np.zeros(self.env.action_space))
        # loop over episodes
        for i_episode in range(1, num_episodes + 1):
            epsilon = 1 - (i_episode / num_episodes)
            episode = []
            state = self.env.reset()
            # generate an episode
            while True:
                greedy_action = Q[state].argmax()
                probs = [epsilon, (1 - epsilon)] if greedy_action == 1 else [(1 - epsilon), epsilon]
                action = np.random.choice(np.arange(2), p=probs)
                next_state, reward, done, info = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state
                if done:
                    break
            # parse episode
            states, actions, rewards = zip(*episode)
            discounts = np.array([self.gamma**i for i in range(len(rewards) + 1)])
            visited = []
            steps = zip(states,actions)
            for i, step in enumerate(steps):
                if self.first_visit and step in visited:
                    continue
                state, action = step
                if self.constant_alpha:
                    Q[state][action] += self.alpha * (
                        sum(rewards[i:] * discounts[:-(1 + i)]) - Q[state][action])
                else:
                    Q[state][action] += (1 / N[state][action]) * (
                        sum(rewards[i:] * discounts[:-(1 + i)]) - Q[state][action])


        # obtain the state-value function
        V = dict((k, np.max(v)) for k, v in Q.items())
        # obtain the policy
        policy = {k: v.argmax() for k, v in Q.items()}

        super().finish_run(V=V, policy=policy)
