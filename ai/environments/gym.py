import gym
import time
import numpy as np
from collections import deque
from math import floor, log
import matplotlib.pyplot as plt
import sys

class GymEnv(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.report_interval = 0
        self.episode = 0
        self.shell_cleanup = False
        self.rewards = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self._cleanup()
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space.n

    def reset(self):
        self._report()
        return self.env.reset()

    def step(self, action):
        step = self.env.step(action)
        next_state, reward, done, info = step
        self.last_reward = reward
        return step

    # def visualize_value(self, value_function):
    #     self._cleanup()
    #     k, v = next(iter(value_function.items()))
    #     if type(k) == tuple:
    #         self._visualize_action_value(value_function)
    #     else:
    #         self._visualize_state_value(value_function)

    def visualize(self):
        self._cleanup()
        # Do visualization

    def _start_graph(self, x=(0,50000), y=(-1,1)):
        plt.close()
        plt.ion()
        fig=plt.figure()
        ax = fig.add_subplot(111, xlim=x, ylim=y, xlabel="test", ylabel="test2")
        # self.detail_data = []
        self.summary_data = []
        # self.detail_line, = ax.plot([], [], 'r-')
        self.summary_line, = ax.plot([], [], 'b-')

    def _update_graph(self):
        # self.detail_line.set_data(np.arange(1, len(self.detail_data)+1), self.detail_data)
        self.summary_line.set_data(
            np.arange(1, len(self.summary_data)+1)*self.report_interval, self.summary_data)
        plt.show()
        plt.pause(0.0001)

    def _report(self):
        """Prints out basic run info.

        Automatically determines the best interval to report at based on the speed of the run,
        and prints the episode number.
        """
        self.episode += 1
        if self.episode == 1:
            self.start = time.time()
        if self.report_interval == 0 and (time.time() - self.start) > 0.1:
            exp = log(self.episode, 10)
            exp = floor(exp)
            self.report_interval = 10**exp
            self.rewards_summary = deque(maxlen=self.report_interval*100)
            self._start_graph()
        if self.report_interval != 0:
            # self.detail_data.append(self.last_reward)
            self.rewards_summary.append(self.last_reward)
            if (self.episode % self.report_interval) == 0:
                self.summary_data.append(np.mean(self.rewards_summary))
                self._update_graph()
                print("\rEpisode: {}  Average Reward: {:.3}".format(
                    self.episode, np.mean(self.rewards_summary)), end="")

    def _cleanup(self):
        if not self.shell_cleanup:
            self.shell_cleanup = True
            print()

    def _visualize_state_value(self, value_function):
        print('State-value function visualization is not implemented for this environment.')

    def _visualize_action_value(self, value_function):
        print('Action-value function visualization is not implemented for this environment.')



# Simple
class Blackjack(GymEnv):
    def __init__(self):
        super().__init__('Blackjack-v0')

    def _visualize_state_value(self, value_function):
        pass

class FrozenLake(GymEnv):
    def __init__(self):
        super().__init__('FrozenLake-v0')

class FrozenLakeLarge(GymEnv):
    def __init__(self):
        super().__init__('FrozenLake8x8-v0')

class Taxi(GymEnv):
    def __init__(self):
        super().__init__('Taxi-v2')

# Classic control
class Acrobot(GymEnv):
    def __init__(self):
        super().__init__('Acrobot-v1')

class CartPole(GymEnv):
    def __init__(self):
        super().__init__('CartPole-v1')

class MountainCar(GymEnv):
    def __init__(self):
        super().__init__('MountainCar-v0')

class MountainCarContinuous(GymEnv):
    def __init__(self):
        super().__init__('MountainCarContinuous-v0')

class Pendulum(GymEnv):
    def __init__(self):
        super().__init__('Pendulum-v0')

# Box2d
class BipedalWalker(GymEnv):
    def __init__(self):
        super().__init__('BipedalWalker-v2')

class BipedalWalkerHardcore(GymEnv):
    def __init__(self):
        super().__init__('BipedalWalkerHardcore-v2')

class CarRacing(GymEnv):
    def __init__(self):
        super().__init__('CarRacing-v0')

class LunarLander(GymEnv):
    def __init__(self):
        super().__init__('LunarLander-v2')

class LunarLanderContinuous(GymEnv):
    def __init__(self):
        super().__init__('LunarLanderContinuous-v2')

# MuJoCo
class Ant(GymEnv):
    def __init__(self):
        super().__init__('Ant-v2')

class HalfCheetah(GymEnv):
    def __init__(self):
        super().__init__('HalfCheetah-v2')

class Hopper(GymEnv):
    def __init__(self):
        super().__init__('Hopper-v2')

class Humanoid(GymEnv):
    def __init__(self):
        super().__init__('Humanoid-v2')

class HumanoidStandup(GymEnv):
    def __init__(self):
        super().__init__('HumanoidStandup-v2')

class InvertedDoublePendulum(GymEnv):
    def __init__(self):
        super().__init__('InvertedDoublePendulum-v2')

class Reacher(GymEnv):
    def __init__(self):
        super().__init__('Reacher-v2')
