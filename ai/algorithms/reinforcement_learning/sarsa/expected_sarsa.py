import numpy as np
from collections import defaultdict
from ai.algorithms.base import BaseAlgorithm

class ExpectedSarsa(BaseAlgorithm):
    def __init__(self, env):
        """Sarsa initialization

        Sarsa learning.

        Arguments:
            env -- An environment to run the algorithm against.
        """
        self.env = env

    def run(self, num_episodes=50000, alpha=.05, gamma=1.0):
        """Runs the Sarsa algorithm

        Runs Sarsa against the initialized evironment.

        Keyword Arguments:
            num_episodes {int} -- The number of episodes to run for. (default: 50000)
            alpha {float} -- The learning rate. (default: 0.05)
            gamma {float} -- The discount rate. (default: 1.0)
        """
        # initialize empty dictionary of arrays
        Q = defaultdict(lambda: np.zeros(self.env.action_space))
        # loop over episodes
        for i_episode in range(1, num_episodes + 1):
            epsilon = 1 / num_episodes
            s0 = self.env.reset()
            policy = self.get_policy(Q, s0, epsilon)
            for i in range(300):
                a0 = np.random.choice(np.arange(self.env.action_space), p=policy)
                s1, r, done, info = self.env.step(a0)
                if done:
                    Q[s0][a0] += alpha * (r + (gamma * 0) - Q[s0][a0])
                    break
                policy = self.get_policy(Q, s1, epsilon)
                Q[s0][a0] += alpha * (r + (gamma * np.dot(Q[s1], policy)) - Q[s0][a0])
                s0 = s1

        return Q

    def get_policy(self, Q, s, epsilon):
        policy = np.ones(self.env.action_space) * epsilon / self.env.action_space
        policy[Q[s].argmax()] = 1 - epsilon + (epsilon / self.env.action_space)
        return policy


# # obtain the estimated optimal policy and corresponding action-value function
# Q_expsarsa = expected_sarsa(env, 10000, 1)

# # print the estimated optimal policy
# policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -
#                             1 for key in np.arange(48)]).reshape(4, 12)
# check_test.run_check('td_control_check', policy_expsarsa)
# print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
# print(policy_expsarsa)

# # plot the estimated optimal state-value function
# plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
