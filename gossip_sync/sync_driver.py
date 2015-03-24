
#!/usr/bin/env python3

#
# Driver program for asynchronous gossip DO for ML.
# 

from mpi4py import MPI
import numpy as np

import matrix_util

# Constants
NUM_LINES_IN_FILE = 100
EPSILON = 1E-7
LEARNING_RATE = 1E-4
MAX_ITERATIONS = 1E6

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
        m = w[0]
        b = w[1]
        w = step_gradient(m, b, training_data)
    return w

def gradient(w, training_data):
    m_gradient = 0.
    b_gradient = 0.
    N = float(len(training_data))
    m = w[0]
    b = w[1]

    for x, y in training_data:
        # w[0] is m and w[1] is b
        m_gradient += (-2/N) * x * (y - (m * x) - b)
        b_gradient += (-2/N) * (y - (m * x) - b) 

    return np.array([m_gradient, b_gradient])

def hasConverged(old_w, w, EPSILON, num_iterations, max_iterations):
    return (abs(w[0] - old_w[0]) < EPSILON and abs(w[1] - old_w[1])) or (num_iterations > max_iterations)

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
        if rank == 0:
            print("\niteration {0}".format(num_iterations))
            print("current model: m: {1}, b: {2}".format(rank,w[0],w[1]))

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

        if rank == 0:
            print("new model: {0}\n".format(w))

        # wait for all nodes to finish this iteration
        comm.Barrier()
        num_iterations += 1

        converged = hasConverged(old_w, w, EPSILON, num_iterations, MAX_ITERATIONS)

if __name__ == '__main__':
    run()
