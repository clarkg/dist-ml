#!/usr/bin/env python3

#
# Driver program for asynchronous gossip DO for ML.
# 

from mpi4py import MPI
import numpy as np
import sys
import time

# Constants
NUM_LINES_IN_FILE = 12
EPSILON = 1E-6
LEARNING_RATE = 1E-3
P = np.array([[(1.0/3.0), (1.0/3.0), 0.0, (1.0/3.0)], [(1.0/3.0), (1.0/3.0),
(1.0/3.0), 0.0], [0.0, (1.0/3.0), (1.0/3.0), (1.0/3.0)], [(1.0/3.0), 0.0, (1.0/3.0), (1.0/3.0)]]) 

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
dataFile = open("test_data.txt", "r")
training_data = []
for line in dataFile:
	if currLine >= startLine and currLine <= endLine:
		x, y = [float(f) for f in line.split(" ")]
		training_data.append(np.array([x, y]))	
	currLine += 1
print training_data

# Init
w = np.array([0.0, 0.0])
z = 1.0

while True:
    # if z is too small, w = c / z will be too big
    if z > EPSILON:
        q = w - LEARNING_RATE * gradient(w, training_data);
        
        # for each node u
        for u in range(0, size):
            if rank != u:
                # Puv is the consensus matrix entry for messages sent from u to v
                Puv = P[rank, u]
                comm.Bsend([np.append(q, z) * Puv, MPI.FLOAT], u, tag = 0)
                
        # Pvv represents the slice of pie that the node itself gets
        Pvv = P[rank, rank]
        c = Pvv * q
        z = Pvv * z

    # while message queue not empty
    while comm.Iprobe(source=MPI.ANY_SOURCE, tag = 0):
        new_data = np.empty(w.size + 1, dtype=np.float64)
        comm.Recv([new_data, MPI.FLOAT], source = MPI.ANY_SOURCE, tag = 0) 
        c += new_data[0:(new_data.size - 1)] # add everything but the last element of new_data
        z += new_data[new_data.size - 1] # add the last element of new_data

    # update w
    w = c / z
    if rank == 0:
        print(w)

# while not converged:

MPI.Finalize()
