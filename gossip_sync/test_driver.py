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
	if rank == 0:
		print("iteration {0}:".format(num_iterations))

	send_data = np.array([num_iterations * 10.0+rank, num_iterations * 10.0+rank])

	# send "send_data" to other nodes
	for u in range(0, size):
		if rank != u:
			comm.Isend([send_data, MPI.FLOAT], u, tag = num_iterations)

	recv_data = [np.empty(send_data.size, dtype=np.float64) for i in range(0, size)]
	recv_data[rank] = send_data

	status = MPI.Status()
	number_of_messages_received = 0

	# comm.Barrier()

	# while number_of_messages_received != (size - 1):
	# 	while not comm.Iprobe(source=MPI.ANY_SOURCE, tag = num_iterations, status = status):
	# 		pass

	# 	# create empty buffer and determine which node the message came from using "status.source"
	# 	new_data = np.empty(send_data.size, dtype=np.float64)
	# 	u = status.Get_source()

	# 	# pop from the message queue
	# 	comm.Irecv([new_data, MPI.FLOAT], source = MPI.ANY_SOURCE, tag = num_iterations) 
	# 	recv_data[u] = new_data
	# 	if rank == 0:
	# 		print("Received {0} from node {1} during node {1}'s iteration {2}".format(new_data,u,status.tag))

	# 	number_of_messages_received += 1

	# without this barrier, only 0 gets sent...
	comm.Barrier()

	while number_of_messages_received != (size - 1):
		#pop from the message queue
		for u in range(0, size):
			new_data = np.empty(send_data.size, dtype=np.float64)
			if rank != u:
				rcvd_from_u = comm.Irecv([new_data, MPI.FLOAT], source = u, tag = num_iterations)

				if rcvd_from_u:
					recv_data[u] = new_data
					number_of_messages_received += 1
					if rank == 0:
						print("Received {0} from node {1}".format(new_data,u))

	num_iterations += 1
	comm.Barrier()

	if rank == 0:
		print("\n")