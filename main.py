import ai.reinforcement_learning.monte_carlo.monte_carlo_prediction as MonteCarloPrediction
import ai.utils.environments.gym as gym

with gym.FrozenLake() as env:
	mc = MonteCarloPrediction.MonteCarloPrediction(env)
	mc.run()
