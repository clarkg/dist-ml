#!/usr/bin/env python3

#
# Driver program for asynchronous gossip DO for ML.
# 

from mpi4py import MPI
import numpy
import time

# stub method definitions
def calculate_mean_squared_error(w, training_data):
	sum = 0
	for x, y in training_data:
		predicted_y = x * w[0] + w[1]	
		sum += (predicted_y - y) ** 2
	return sum / len(training_data)	
	
def receive_new_values():
	while not comm.Iprobe(source=MPI.ANY_SOURCE, tag=0):
		time.sleep(.001)

	new_data = numpy.empty(100, dtype='i')
	comm.Recv([new_data, MPI.INT], source=MPI.ANY_SOURCE, tag=0)
	#print("Rank {0} received".format(rank))  

def update(w,rcvd_values,MSE):
	return w

def send_new_w(w):
   	data = numpy.arange(100, dtype='i')

	for i in range(0, size):
		if rank != i:
			comm.Bsend([data, MPI.INT], i, tag=0)
		# 	print("Rank {0} bsend to {1}".format(rank, i)) 
 
# adapted from http://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
def stepGradient(m_current, b_current, points, learningRate):
	b_gradient = 0
    	m_gradient = 0
    	N = float(len(points))
	for i in range(0, len(points)):
        	b_gradient += -(2./N) * (points[i][1] - ((m_current*points[i][0]) + b_current))
        	m_gradient += -(2./N) * points[i][0] * (points[i][1] - ((m_current * points[i][0]) + b_current))
	if rank == 3:
		print(m_gradient)
    	new_b = b_current - (learningRate * b_gradient)
    	new_m = m_current - (learningRate * m_gradient)
    	return new_m, new_b

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# initial values of w, convergence conditions
# w[0] is m, w[1] is m where y = mx+b
w = numpy.array([1., 1.])

max_iterations = 20000
number_of_iterations = 0

converged = False
MSE = 0 # mean squared error

# Partition data
NUM_LINES_IN_FILE = 12
linesPerTask = NUM_LINES_IN_FILE / size
startLine = rank * linesPerTask
endLine = (rank + 1) * linesPerTask - 1

currLine = 0
dataFile = open("test_data.txt", "r")
training_data = []
for line in dataFile:
	if currLine >= startLine and currLine <= endLine:
		x, y = [float(f) for f in line.split(" ")]
		training_data.append((x, y))	
	currLine += 1
print training_data

while not converged:
	# compute new w using training data
	new_m, new_b = stepGradient(w[0], w[1], training_data, 0.005)
	w[0] = new_m
	w[1] = new_b	
	if rank == 3:
		print("Rank {0} new m: {1} new b: {2}".format(rank, w[0], w[1]))
		
	# send new w to all neighbors
	send_new_w(w)
	# print("Done sending values")
	# receive new values (flush queue)
	rcvd_values = receive_new_values()

	# print("Done receiving values")
	# update w using received values

	# recalculate parameter, check for convergence
	number_of_iterations += 1
	if number_of_iterations == max_iterations:
		converged = True

print ("Done!")
MPI.Finalize()
