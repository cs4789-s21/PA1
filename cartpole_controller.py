import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, x, u):
        """
        Cost function of the env.
        It sets the state of environment to `x` and then execute the action `u`, and
        then return the cost. 
        Parameter:
            x (1D numpy array) with shape (4,) 
            u (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert x.shape == (4,)
        assert u.shape == (1,)
        env = self.env
        env.reset(state=x)
        env.step(u)
        observation, cost, done, info = env.step(u)
        return cost

    def f(self, x, u):
        """
        State transition function of the environment.
        Return the next state by executing action `u` at the state `x`
        Parameter:
            x (1D numpy array) with shape (4,) 
            u (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert x.shape == (4,)
        assert u.shape == (1,)
        env = self.env
        env.reset(state = x)
        next_observation, cost, done, info = env.step(u)
        return next_observation


    def compute_local_policy(self, x_s, u_s, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (x_s, u_s). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            x_s (numpy array) with shape (4,)
            u_s (numpy array) with shape (1,)
        return 
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        #TODO

        return [(np.zeros((1,4)), np.zeros(1)) for _ in range(T)]

