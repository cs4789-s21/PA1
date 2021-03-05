import gym
import numpy as np
from cartpole_controller import LocalLinearizationController
from env.cartpole_control_env import CartPoleControlEnv

init_states = [np.array([0, 0, 0, 0]),
        np.array([0, 0, 0.2, 0]),
        np.array([0, 0, 0.4, 0]),
        np.array([0, 0, 0.6, 0]),
        np.array([0, 0, 0.8, 0]),
        np.array([0, 0, 1.0, 0]),
        np.array([0, 0, 1.2, 0]),
        np.array([0, 0, 1.4, 0])]

def test(init_state, x_s, u_s, T=500, num_episodes=100):
    env = gym.make("env:CartPoleControlEnv-v0")
    controller = LocalLinearizationController(env)
    policies = controller.compute_local_policy(x_s, u_s, T)

    # For testing, we use a noisy environment which adds small Gaussian noise to
    # state transition. Your controller only need to consider the env without noise.
    env = gym.make("env:NoisyCartPoleControlEnv-v0")
    total_cost = 0
    for _ in range(num_episodes):
        observation = env.reset(state=init_state)
        for (K,k) in policies:
            action = (K @ observation + k)
            observation, cost, done, info = env.step(action)
            total_cost += cost
            if done:
                break
        env.close()

    return total_cost / num_episodes

def main():
    x_s = np.array([0, 0, 0, 0])
    u_s = np.array([0])
    for i, s in enumerate(init_states):
        print("case {} avergae cost:".format(i), test(s, x_s, u_s))

if __name__ == "__main__":
    main()
