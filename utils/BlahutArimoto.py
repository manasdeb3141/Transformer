import numpy as np


# Blahut-Arimoto algorithm for computing the capacity of the channel given the 
# channel input-output transition probability matrix
def blahut_arimoto_capacity(p_y_x: np.ndarray,  thresh: float = 1e-12, max_iter: int = 1e3) -> tuple:
    '''
    Computes the maximum capacity of the channel from the channel probability matrix p(Y|X)
    The Blahut-Arimoto algorithm is iterative and the iteration stops when the difference 
    between the input probability distribution r(x) of the previous iteration and the current
    iteration is below the specified threshold OR if the number of iterations is equal to the
    max iterations specified

    Parameters:
        p_y_x       : Channel transition matrix p(Y|X). This is of shape [m,n] where m is the 
                      number of possible input values and n is the number of possible output 
                      values of the channel
                      
        thresh      : Threshold to stop the iteration. The iteration stops if ||r_curr - r_new|| < threshold 

        max_iter    : Maxim number of iterations

    Returns:
        capacity    : The capacity of the channel in bits 
        r           : The input distribution that achieves this capacity
    '''

    # Sanity check that the channel probability matrix is not passed in
    # as a row vector
    assert (p_y_x.shape[0] > 1)

    # Each row of the channel transition matrix should sum up to 1
    # since each row is p(X=x, Y)
    assert np.abs(p_y_x.sum(axis=1).mean() - 1) < 1e-6

    # The number of possible input values: size of |X|
    m = p_y_x.shape[0]

    # The number of possible output values: size of |Y|
    n = p_y_x.shape[1]

    # Initialize the input distribution r(x) uniformly
    r = np.ones((1, m)) / m

    # Compute the r(x) that maximizes the capacity
    for iteration in range(int(max_iter)):
        # Step 1 : Compute q(x|y) from r(x) and p(y|x)
        #
        # For each row of the matrix p_y_x
        #    Each element of row vector r is multiplied
        #    by the corresponding element of the row of 
        #    matrix p_y_x. Therefore q is a matrix of the
        #    same size as p_y_x
        #
        # for i in range(p_y_x.shape[0]):
        #    for j in range(p_y_x.shape[1]):
        #        q[i, j] = r[j] * p_y_x[i, j]
        #
        q = r.T * p_y_x
        q = q / np.sum(q, axis=0)

        # Step 2: Compute r_new(x)
        r_new = np.prod(np.power(q, p_y_x), axis=1)
        r_new = r_new / np.sum(r_new)

        # delta = ||r_new - r||
        delta = np.linalg.norm(r_new - r)

        # Assign r(x) = r_new(x) for the next
        # iteration 
        r = r_new

        if delta < thresh:
            break

    # Calculate the capacity
    r = r.flatten()
    capacity = 0
    for i in range(m):
        if r[i] > 0:
            capacity += np.sum(r[i] * p_y_x[i, :] * np.log((q[i, :] / r[i]) + 1e-16))

    # Return the capacity in bits
    capacity = capacity / np.log(2)

    return capacity, r

    
import numpy as np


# Blahut-Arimoto algorithm for computing the distortion of the channel
# given the input distribution
def blahut_arimoto_distortion(p_x, D_max, beta, thresh=1e-6, max_iter=1000):
    """
    Blahut-Arimoto algorithm for rate-distortion optimization.

    Parameters:
        p_x:      Probability distribution of the source symbols.
        D_max:    Maximum distortion matrix.
        beta:     Lagrange multiplier controlling the rate-distortion trade-off.
        thresh:   Convergence threshold
        max_iter: Maximum number of iterations.

    Returns:
        Q:        Transition probability matrix
        R:        Rate in bits per channel use
        D:        Distortion matrix
    """

    n = len(p_x)

    # Initialize the conditional probability matrix to a uniform 
    # distribution
    Q = np.ones((n, n)) / n

    for _ in range(max_iter):
        #
        # Step 1: calculate the input distribution r(x) that minimizes the
        #         mutual information
        #
        r = np.matmul(p_x, Q)

        #
        # Step 2: minimize Q(x_hat|x)
        #
        Q_new = r * np.exp(-beta * D_max)
        Q_new = Q_new / np.sum(Q_new, axis=1, keepdims=True)

        delta = np.abs(Q_new - Q)

        # Set the value of Q(x_hat|x) for the next iteration
        Q = Q_new

        if np.max(delta) < thresh:
            break

    R = np.matmul(p_x, Q*np.log(Q/np.expand_dims(r, 0))).sum()/np.log(2)
    D = np.matmul(p_x, (Q * D_max)).sum()

    return Q, R, D