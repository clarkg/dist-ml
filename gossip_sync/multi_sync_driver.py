
#!/usr/bin/env python3

#
# Driver program for asynchronous gossip DO for ML.
# 

from mpi4py import MPI
import numpy as np

import matrix_util

# Constants
NUM_LINES_IN_FILE = 100
EPSILON = 1E-12
LEARNING_RATE = 1E-4
MAX_ITERATIONS = 1E4

DATA_LOCATION = "../line_data2.txt"

# P is the consensus matrix
P = np.array([[(1.0/3.0), (1.0/3.0), 0.0, (1.0/3.0)], [(1.0/3.0), (1.0/3.0),
(1.0/3.0), 0.0], [0.0, (1.0/3.0), (1.0/3.0), (1.0/3.0)], [(1.0/3.0), 0.0, (1.0/3.0), (1.0/3.0)]]) 

# 1/3 1/3 0 1/3
# 1/3 1/3 1/3 0
# 0 1/3 1/3 1/3
# 1/3 0 1/3 1/3

def gradient_descent_runner(training_data, w, LEARNING_RATE, num_iterations):
    for i in range(num_iterations):
        m = w[:-1]
        b = w[-1]
        w = step_gradient(w, training_data)
    return w

def gradient(w, training_data):
    """ w : vector of (d + 1) elements m::b

        m : vector of d elements [m_1, m_2, ..., m_d]
        b : scalar

        so w has form [m_1, m_2, ..., m_d, b]

        training_data : list of (d + 1)-sized vectors x::y
                         where x is an d-dimensional vector
                        and y is a scalar value
        n : dimensions of vectors

        output : w
                where grad_m is a d-dimensional vector
                and grad_b is a scalar"""
    assert w.ndim == 1
    
    m_gradient = np.zeros(w.size - 1, dtype = np.float)
    b_gradient = 0.
    N = float(len(training_data)) # number of training data pairs
    m = w[:-1]
    b = w[-1] 
    d =  m.size

    for pair in training_data:
        x = pair[:-1]
        y = pair[-1]

        common_loss_sum = (-2. / N) * (y - np.dot(m, x) - b)
        b_gradient += common_loss_sum
        for i in range(d):
            m_gradient[i] +=  x[i] * common_loss_sum

    return np.append(m_gradient, b_gradient)

def computeError(old_w, w):
    """ old_w : vector of d elements
        w : vector of d elements

        output : Error between old_w and w """
    d = w.size

    error = 0.
    for i in range(d):
        error += ((old_w[i] - w[i]) ** 2)
    return error

def hasConverged(old_w, w, EPSILON, num_iterations, max_iterations):
    return computeError(old_w, w) < EPSILON or (num_iterations > max_iterations)

def run():
    assert matrix_util.isSquare(P)
    assert matrix_util.isColumnStochastic(P)
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(P.size)
   
    # Make sure P has the correct dimensions given number of processes 
    assert P.shape[0] == size

    # Partition data
    lines_per_task = NUM_LINES_IN_FILE / size
    start_line = rank * lines_per_task
    end_line = (rank + 1) * lines_per_task - 1

    curr_line = 0
    data_file = open(DATA_LOCATION, "r")
    training_data = []
    for line in data_file:
        if curr_line >= start_line and curr_line <= end_line:
            x, y = [float(f) for f in line.split(" ")]
            training_data.append(np.array([x, y]))	
        curr_line += 1

    # Init
    w = np.array([0.0, 0.0]) # w will represent the current guess
    q = np.array([0.0, 0.0]) # q will represent the piece that each node sends to its neighbors

    num_iterations = 0

    converged = False
    while not converged:

        old_w = w

        q = w - LEARNING_RATE * gradient(w, training_data)

        # send q to other nodes
        for u in range(0, size):
            if rank != u:
                comm.Isend([q, MPI.FLOAT], u, tag = num_iterations)

        # collect q from other nodes
        q_u = [np.empty(q.size, dtype=np.float64) for i in range(0, size)]
        q_u[rank] = q

        status = MPI.Status()
        number_of_messages_received = 0

        # without this barrier, only 0 gets sent...
        comm.Barrier()

        while number_of_messages_received != (size - 1):
            # pop from the message queue
            for u in range(0, size):
                new_data = np.empty(q.size, dtype=np.float64)
                if rank != u:
                    rcvd_from_u = comm.Irecv([new_data, MPI.FLOAT], source = u, tag = num_iterations)

                    if rcvd_from_u:
                        q_u[u] = new_data
                        number_of_messages_received += 1
                        

        
        w = np.array([0., 0.]) # zero out w
        for u in range(0, size):
            Puv = P[rank, u]
            w += q_u[u] * Puv

        # wait for all nodes to finish this iteration
        comm.Barrier()
        num_iterations += 1

        converged = hasConverged(old_w, w, EPSILON, num_iterations, MAX_ITERATIONS)

    print("Final model: {0}\n".format(w))

if __name__ == '__main__':
    run()
