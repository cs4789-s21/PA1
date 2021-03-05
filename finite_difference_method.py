import numpy as np

def gradient(f, x, delta = 1e-5):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    #TODO
    pass

def jacobian(f, x, delta = 1e-5):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    #TODO
    pass

def hessian(f, x, delta = 1e-5):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    #TODO
    pass

def test():
    """
    Run tests on the above functions against some known case.
    """
    Q = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]], dtype=np.float32)
    q = np.array([10, 11, 12], dtype=np.float32)
    x_s = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    f1 = lambda x: Q @ (x - x_s) + np.ones(3)
    f2 = lambda x: (x - x_s) @ Q @ (x - x_s) + q @ (x - x_s) + 1
    assert np.allclose(jacobian(f1, np.array(x_s)), Q)
    assert np.allclose(gradient(f2, np.array(x_s)), q)
    assert np.allclose(hessian(f2, np.array(x_s)), Q+Q.T)


if __name__ == "__main__":
    test()
