
#!/usr/bin/env python3

#
# Driver program for asynchronous gossip DO for ML.
# 

from mpi4py import MPI
import numpy as np
import sys
import time

# Constants
NUM_LINES_IN_FILE = 100000
EPSILON = 1E-6
LEARNING_RATE = 1E-9
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
    for x, y in training_data:
        # w[0] is m and w[1] is b
        m_gradient += -2 * (y - (w[0] * x) + w[1]) * x
        b_gradient += -2 * (y - (w[0] * x) + w[1]) 

    m_gradient /= len(training_data)
    b_gradient /= len(training_data)

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
dataFile = open("../line_data.txt", "r")
training_data = []
for line in dataFile:
	if currLine >= startLine and currLine <= endLine:
		x, y = [float(f) for f in line.split(" ")]
		training_data.append(np.array([x, y]))	
	currLine += 1

# Init
w = np.array([0.0, 0.0])
q = 0.0

print("{0}\n".format(rank))
print("{0}\n".format(P))

num_iterations = 0

while num_iterations < 10:
    q = w - (LEARNING_RATE * gradient(w, training_data))

    # send q to other nodes
    for u in range(0, size):
        if rank != u:
            comm.Isend([np.append(q, z) * Puv, MPI.FLOAT], u, tag = 0)

    # collect q from other nodes
    q_u = [np.empty(q.size, dtype=np.float64) for i in range(0, size)]
    q_u[rank] = q

    status = MPI.Status()
    number_of_messages_received = 0
    while comm.Iprobe(source=MPI.ANY_SOURCE, tag = 0, status = status) and number_of_messages_received != (size - 1):
        # create empty buffer and determine which node the message came from
        # (so we know which Puv to apply to it)
        new_data = np.empty(w.size, dtype=np.float64)
        u = status.source
        
        # pop from the message queue
        comm.Recv([new_data, MPI.FLOAT], source = MPI.ANY_SOURCE, tag = 0) 
        q_u[u] = new_data

    # wait for all nodes to finish this iteration
    comm.barrier()
    num_iterations += 1
