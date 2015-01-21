#
# Driver program for asynchronous gossip DO for ML.
# 

from mpi4py import MPI

# stub method definitions
def calculate_mean_squared_error(w):
	return 1

def receive_new_values():
	return 1

def update(w,rcvd_values,MSE):
	return w

def send_new_w(w):
	return

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# initial values of w, convergence conditions
m0 = 2
b0 = 3

max_iterations = 10
number_of_iterations = 0
w = {'m' : m0, 'b' : b0}
converged = False
MSE = 0 # mean squared error

while not converged:
	# calculate error function of w on data
	MSE = calculate_mean_squared_error(w)

	# receive new values (flush queue)
	rcvd_values = receive_new_values()

	# update w
	w = update(w,rcvd_values,MSE)

	# send new w to all neighbors
	send_new_w(w)

	# recalculate parameter, check for convergence
	number_of_iterations += 1
	if number_of_iterations == max_iterations:
		converged = True

print w['m']
print w['b']
