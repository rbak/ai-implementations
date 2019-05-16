import gym
import time
from math import floor, log

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
        return self.env.step(action)

    def visualize_value(self, value_function):
        self._cleanup()
        k, v = next(iter(value_function.items()))
        if type(k) == tuple:
            self._visualize_action_value(value_function)
        else:
            self._visualize_state_value(value_function)

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
        if self.report_interval != 0 and (self.episode % self.report_interval) == 0:
            print("\rEpisode: {}".format(self.episode), end="")

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