
#!/usr/bin/env python3

#
# Driver program for asynchronous gossip DO for ML.
# 

from mpi4py import MPI
import numpy as np
import sys
import time

# Constants
NUM_LINES_IN_FILE = 100
EPSILON = 1E-7
LEARNING_RATE = 1E-4
P = np.array([[(1.0/3.0), (1.0/3.0), 0.0, (1.0/3.0)], [(1.0/3.0), (1.0/3.0),
(1.0/3.0), 0.0], [0.0, (1.0/3.0), (1.0/3.0), (1.0/3.0)], [(1.0/3.0), 0.0, (1.0/3.0), (1.0/3.0)]]) 
MAX_ITERATIONS = 1E6

# 1/3 1/3 0 1/3
# 1/3 1/3 1/3 0
# 0 1/3 1/3 1/3
# 1/3 0 1/3 1/3

def isStochastic(vector):
    """Returns true if numpy vector is stochastic"""
	
    # if not equal to 1 within machine precision
    if abs(np.sum(vector) - 1) < sys.float_info.epsilon:
        return True

def isColumnStochastic(P):
    """Returns true if 2D array P is column stochastic"""
    assert P.ndim >= 2
    # iterate over columns by iterating over rows of the transpose
    for column in P.T:
        if not isStochastic(column):
            return False
    return True	

def isRowStochastic(P):
    """Returns true if 2D array P is column stochastic"""
    assert P.ndim >= 2
	
    for row in P:
        if not isStochastic(row):
            return False
    return True	

def isDoublyStochastic(P):	
    """Returns true if 2D array P is doubly stochastic"""
    assert P.ndim >= 2
    return isColumnStochastic(P) and isRowStochastic(P)

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
        # if rank == 0: print("x: {0}, y: {1}".format(x,y))
        # w[0] is m and w[1] is b
        m_gradient += (-2/N) * x * (y - (m * x) - b)
        b_gradient += (-2/N) * (y - (m * x) - b) 

    # new_m = m - LEARNING_RATE * m_gradient
    # new_b = b - LEARNING_RATE * b_gradient

    # return np.array([new_m, new_b])

    return np.array([m_gradient, b_gradient])

def hasConverged(old_w, w, EPSILON, num_iterations, max_iterations):
    return (abs(w[0] - old_w[0]) < EPSILON and abs(w[1] - old_w[1])) or (num_iterations > max_iterations)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Partition data
linesPerTask = NUM_LINES_IN_FILE / size
startLine = rank * linesPerTask
endLine = (rank + 1) * linesPerTask - 1

currLine = 0
dataFile = open("../line_data2.txt", "r")
training_data = []
for line in dataFile:
	if currLine >= startLine and currLine <= endLine:
		x, y = [float(f) for f in line.split(" ")]
		training_data.append(np.array([x, y]))	
	currLine += 1

# Init
w = np.array([0.0, 0.0])
q = np.array([0.0, 0.0])

# print("rank {0} beginning\n".format(rank))

# starting_b = 0
# starting_m = 0
# [m,b] = gradient_descent_runner(training_data, w, LEARNING_RATE, 100000)
# print("After 100000 iterations, b = {0}, m = {1}".format(b,m))

num_iterations = 0


converged = False
while not converged:

    old_w = w
    if rank == 0:
        print("\niteration {0}".format(num_iterations))
        print("current model: m: {1}, b: {2}".format(rank,w[0],w[1]))

    # if rank == 0:
    #     grad_w = gradient(w, training_data)
    #     print("node {0} computing grad w...".format(rank))
    #     print("grad m at node {0} = {1}".format(rank,grad_w[0]))
    #     print("grad b at node {0} = {1}".format(rank,grad_w[1]))
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

    # while number_of_messages_received != (size - 1):
    #     while not comm.Iprobe(source=MPI.ANY_SOURCE, tag = num_iterations, status = status):
    #         # create empty buffer and determine which node the message came from
    #         # (so we know which Puv to apply to it)
    #         pass

    #     new_data = np.empty(w.size, dtype=np.float64)
    #     u = status.source
            
    #     # pop from the message queue
    #     comm.Irecv([new_data, MPI.FLOAT], source = MPI.ANY_SOURCE, tag = num_iterations) 
    #     q_u[u] = new_data
    #     if rank == 0:
    #         print("Received {0} from node {1} during node {1}'s iteration {2}".format(new_data,u,status.tag))

    #     number_of_messages_received += 1

    # without this barrier, only 0 gets sent...
    comm.Barrier()

    while number_of_messages_received != (size - 1):
        #pop from the message queue
        for u in range(0, size):
            new_data = np.empty(q.size, dtype=np.float64)
            if rank != u:
                rcvd_from_u = comm.Irecv([new_data, MPI.FLOAT], source = u, tag = num_iterations)

                if rcvd_from_u:
                    q_u[u] = new_data
                    number_of_messages_received += 1
                    

    
    w = np.array([0., 0.])
    for u in range(0, size):
        Puv = P[rank, u]
        w += q_u[u] * Puv

    if rank == 0:
        print("new model: {0}\n".format(w))

    # wait for all nodes to finish this iteration
    comm.Barrier()
    num_iterations += 1

    converged = hasConverged(old_w, w, EPSILON, num_iterations, MAX_ITERATIONS)
