import numpy as np
def lqr(A, B, m, Q, R, M, q, r, T):
    """
    Compute optimal policise by solving
    argmin_{\pi_0,...\pi_{T-1}} \sum_{t=0}^{T-1} x_t^T Q x_t + u_t^T R u_t + x_t^T M u_t + q^T x_t + r^T u_t
    subject to x_{t+1} = A x_t + B u_t + m, u_t = \pi_t(x_t)
    
    Let the shape of x_t be (N_x,), the shape of u_t be (N_u,)
    Let optimal \pi*_t(x) = K_t x + k_t
    
    Parameters:
    A (2d numpy array): A numpy array with shape (N_x, N_x)
    B (2d numpy array): A numpy array with shape (N_x, N_u)
    m (1d numpy array): A numpy array with shape (N_x,)
    Q (2d numpy array): A numpy array with shape (N_x, N_x)
    R (2d numpy array): A numpy array with shape (N_u, N_u)
    M (2d numpy array): A numpy array with shape (N_x, N_u)
    q (1d numpy array): A numpy array with shape (N_x,)
    r (1d numpy array): A numpy array with shape (N_u,)
    T (int): The number of total steps in finite horizon settings

    Returns:
        ret (list): A list, [(K_0, k_0), (K_1, k_1), ..., (K_{T-1}, k_{T-1})]
        and the shape of K_t is (N_u, N_x), the shape of k_t is (N_u,)
    """
    #TODO
    pass

