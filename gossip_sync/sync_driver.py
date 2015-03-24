
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
EPSILON = 1E-6
LEARNING_RATE = 1E-3
P = np.array([[(1.0/3.0), (1.0/3.0), 0.0, (1.0/3.0)], [(1.0/3.0), (1.0/3.0),
(1.0/3.0), 0.0], [0.0, (1.0/3.0), (1.0/3.0), (1.0/3.0)], [(1.0/3.0), 0.0, (1.0/3.0), (1.0/3.0)]]) 

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


def gradient(w, training_data):
    m_gradient = 0.
    b_gradient = 0.
    N = len(training_data)
    m = w[0]
    b = w[1]

    for x, y in training_data:
        # if rank == 0: print("x: {0}, y: {1}".format(x,y))
        # w[0] is m and w[1] is b
        m_gradient += (-2/N) * x * (y - (m * x) - b)
        b_gradient += (-2/N) * (y - (m * x) - b) 

    # m_gradient /= len(training_data)
    # b_gradient /= len(training_data)

    return np.array([m_gradient, b_gradient])

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

num_iterations = 0

while num_iterations < 10:
    if rank == 0:
        print("\niteration {0}".format(num_iterations))
        print("current model: m: {1}, b: {2}".format(rank,w[0],w[1]))

    # if rank == 0:
    #     grad_w = gradient(w, training_data)
    #     print("node {0} computing grad w...".format(rank))
    #     print("grad m at node {0} = {1}".format(rank,grad_w[0]))
    #     print("grad b at node {0} = {1}".format(rank,grad_w[1]))
    q = w - (LEARNING_RATE * gradient(w, training_data))

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
                    if rank == 0:
                        print("Node {1}'s q: {0}".format(new_data,u))

    print("Node 0's q: {}".format(q))
    for u in range(0, size):
        Puv = P[rank, u]
        w += q_u[u] * Puv

    if rank == 0:
        print("new model: {0}\n".format(w))

    # wait for all nodes to finish this iteration
    comm.barrier()
    num_iterations += 1
