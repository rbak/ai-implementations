import numpy as np
from collections import defaultdict
from ai.algorithms.base import BaseAlgorithm

class Sarsa(BaseAlgorithm):
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
        # initialize
        Q = defaultdict(lambda: np.zeros(self.env.action_space))
        # loop over episodes
        for i_episode in range(1, num_episodes + 1):
            epsilon = 1 / num_episodes
            s0 = self.env.reset()
            policy = self.get_policy(Q, s0, epsilon)
            a0 = np.random.choice(np.arange(self.env.action_space), p=policy)
            for i in range(300):
                s1, r, done, info = self.env.step(a0)
                if done:
                    Q[s0][a0] += alpha * (r + (gamma * 0) - Q[s0][a0])
                    break
                policy = self.get_policy(Q, s1, epsilon)
                a1 = np.random.choice(np.arange(self.env.action_space), p=policy)
                Q[s0][a0] += alpha * (r + (gamma * Q[s1][a1]) - Q[s0][a0])
                s0, a0 = s1, a1

        return Q

        policy = {k: v.argmax() for k, v in Q.items()}
        # obtain the corresponding state-value function
        V = dict((k, np.max(v)) for k, v in Q.items())
        # self.env.visualize_value(value_function)

    def get_policy(self, Q, s, epsilon):
        policy = np.ones(self.env.action_space) * epsilon / self.env.action_space
        policy[Q[s].argmax()] = 1 - epsilon + (epsilon / self.env.action_space)
        return policy


# # obtain the estimated optimal policy and corresponding action-value function
# Q_sarsa = sarsa(env, 5000, .01)

# # print the estimated optimal policy
# policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4, 12)
# check_test.run_check('td_control_check', policy_sarsa)
# print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
# print(policy_sarsa)

# # plot the estimated optimal state-value function
# V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
# plot_values(V_sarsa)
