from mpi4py import MPI
import numpy as np
import sys
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_iterations = 0
while num_iterations < 10:
	send_data = 

    # send "send_data" to other nodes
    for u in range(0, size):
        if rank != u:
            comm.Isend([send_data, MPI.FLOAT], u, tag = num_iterations)

    num_iterations += 1

