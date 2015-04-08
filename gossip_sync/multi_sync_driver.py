#!/usr/bin/env python3

#
# Driver program for synchronous gossip DO for ML.
#

from mpi4py import MPI
import getopt
import numpy as np
import sys

import consensus_matrix as consensus
import matrix_util

# Constants
DEF_NUM_LINES = int(1E2)
DEF_EPSILON = 1E-12
DEF_LEARN_RATE = 1E-4
DEF_MAX_ITER = 1E4
DEF_DIM = 2
DEF_DATA_LOC = "../test_data/multivariate_line_data_d2_n100.txt"
DEF_RING = 0
DEF_TOPOLOGY = 1

HELP_MESSAGE = "mpirun [<MPI OPTIONS>] <python3 | python> multi_sync_driver.py  [-i <inputDataFile>] [-d <dimensionality>] [-n <num_iterations>] [-e <epsilon>] [-r <learning_rate>] [-l <num_lines_in_file>]"

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

    m_gradient = np.zeros(w.size - 1, dtype=np.float)
    b_gradient = 0.
    N = float(len(training_data))  # number of training data pairs
    m = w[:-1]
    b = w[-1]
    d = m.size

    for pair in training_data:
        x = pair[:-1]
        y = pair[-1]

        common_loss_sum = (-2. / N) * (y - np.dot(m, x) - b)
        b_gradient += common_loss_sum
        for i in range(d):
            m_gradient[i] += x[i] * common_loss_sum

    return np.append(m_gradient, b_gradient)


def partitionData(data_loc, num_lines, num_processes, dim, rank):
    """ data_loc : Path to input data file
        num_lines : Number of lines in file
        num_processes : Number of processes amongst which to partition data
        dim : Dimensionality 

        output : training_data
                which is a list of numpy arrays where
                the last element is a scalar output y and the 
                first element is a vector input [x_0, x_1, ..., x_(dim - 1)]"""
    # Partition data
    lines_per_task = num_lines / num_processes
    start_line = rank * lines_per_task
    end_line = (rank + 1) * lines_per_task - 1

    curr_line = 0
    data_file = open(data_loc, "r")
    training_data = []
    for line in data_file:
        if curr_line >= start_line and curr_line <= end_line:
            new_datum = np.array([float(f) for f in line.split(" ")])

            if new_datum.size != dim + 1:
                raise ValueError(
                    "Dimensionality {0} does not match input file {1} which instead matches dimensionality {2}.".format(
                        dim, data_loc, new_datum.size - 1))
            training_data.append(new_datum)
        curr_line += 1
    return training_data


def computeError(old_w, w):
    """ old_w : vector of d elements
        w : vector of d elements

        output : Error between old_w and w """
    d = w.size

    error = 0.
    for i in range(d):
        error += ((old_w[i] - w[i]) ** 2)
    return error

def hasConverged(old_w, w, epsilon, num_iter, max_iter):
    return computeError(old_w, w) < epsilon or (num_iter > max_iter)

def getConstants(argv):
    dim = DEF_DIM
    epsilon = DEF_EPSILON
    data_loc = DEF_DATA_LOC
    num_lines = DEF_NUM_LINES
    max_iter = DEF_MAX_ITER
    learn_rate = DEF_LEARN_RATE
    topology = DEF_RING

    try:
        opts, args = getopt.getopt(argv, "hd:e:i:l:n:r",
                                   ["help", "dim=", "epsilon=", "input_file=",
                                    "num_lines=", "num_iterations=",
                                    "learn_rate="])
    except getopt.GetoptError:
        print(HELP_MESSAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "help"):
            print(HELP_MESSAGE)
            sys.exit()
        elif opt in ("-d", "dim"):
            dim = int(arg)
            assert dim > 0
        elif opt in ("-e", "epsilon"):
            epsilon = float(arg)
            assert epsilon > 0.
        elif opt in ("-i", "--input_file"):
            data_loc = arg
        elif opt in ("-l", "--num_lines"):
            num_lines = int(arg)
            assert num_lines > 0
        elif opt in ("-n", "--num_iterations"):
            max_iter = int(arg)
            assert num_iterations > 0
        elif opt in ("-r", "--learn_rate"):
            learn_rate = float(arg)
            assert learning_rate > 0.
        elif opt in ("-u", "--usr_def_topology")
            topology = DEF_TOPOLOGY

    return (dim, epsilon, data_loc, num_lines, max_iter, learn_rate, topology)


def initMPI():
    comm = MPI.COMM_WORLD
    return (MPI.COMM_WORLD, comm.Get_rank(), comm.Get_size())


def run(argv):
    (dim, epsilon, data_loc, num_lines, max_iter,
     learn_rate, topology) = getConstants(argv)

    (comm, rank, size) = initMPI()
    if topology == DEF_TOPOLOGY:
         A = consensus.usr_def_adjacency_matrix()
    else:
         A = consensus.adjacency_matrix(size)

    P = consensus.generate_consensus_matrix(A)
    assert matrix_util.isSquare(P)
    assert matrix_util.isColumnStochastic(P)

    # Make sure P has the correct dimensions given number of processes
    assert P.shape[0] == size

    # Partition data
    training_data = partitionData(data_loc, num_lines, size, dim, rank)

    # Init
    w = np.zeros(dim + 1, dtype=np.float)  # w will represent the current guess
    q = np.zeros(
        dim + 1,
        dtype=np.float
    )  # q will represent the piece that each node sends to its neighbors

    num_iterations = 0

    converged = False
    old_w = w
    while not converged:
        old_w = w
        q = w - learn_rate * gradient(w, training_data)

        # send q to other nodes
        for u in range(0, size):
            if rank != u and P[rank, u] != 0:
                comm.Isend([q, MPI.FLOAT], u, tag=num_iterations)

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
                if rank != u and P[rank, u] != 0:
                    rcvd_from_u = comm.Irecv([new_data, MPI.FLOAT],
                                             source=u,
                                             tag=num_iterations)

                    if rcvd_from_u:
                        q_u[u] = new_data
                        number_of_messages_received += 1

        w = np.zeros(dim + 1)  # zero out w
        for u in range(0, size):
            if P[rank, u] != 0:
                Puv = P[rank, u]
                w += q_u[u] * Puv

        # wait for all nodes to finish this iteration
        comm.Barrier()
        num_iterations += 1

        converged = hasConverged(old_w, w, epsilon, num_iterations, max_iter)

    if rank == 0:
        print("Final model: {0}".format(w))
        print("Error : {0}".format(computeError(old_w, w)))
        print("Number of iterations : {0}\n".format(num_iterations - 1))


if __name__ == '__main__':
    run(sys.argv[1:])
