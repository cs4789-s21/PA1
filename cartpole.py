import gym
import numpy as np
from cartpole_controller import LocalLinearizationController
from gym import wrappers

video_path = "./gym-results"
x_init = np.array([0, 0, 0, 0])
x_s = np.array([0, 0, 0, 0], dtype=np.double)
u_s = np.array([0], dtype=np.double)
T = 500

env = gym.make("env:CartPoleControlEnv-v0")
controller = LocalLinearizationController(env)
policies = controller.compute_local_policy(x_s, u_s, T)

# For testing, we use a noisy environment which adds small Gaussian noise to
# state transition. Your controller only need to consider the env without noise.
env = gym.make("env:NoisyCartPoleControlEnv-v0")

env = wrappers.Monitor(env, video_path, force = True)
total_cost = 0
observation = env.reset(state = x_init)

for (K,k) in policies:
    env.render()
    action = (K @ observation + k)
    observation, cost, done, info = env.step(action)
    total_cost += cost
    if done: # When the state is out of the range, the cost is set to inf and done is set to True
        break
env.close()
print("cost = ", total_cost)
