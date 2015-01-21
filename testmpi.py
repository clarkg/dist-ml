from mpi4py import MPI

MASTER = 0
NUM_LINES_IN_FILE = 15

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = None
if rank != MASTER:
	linesPerTask = NUM_LINES_IN_FILE / (size - 1)
	startLine = (rank - 1) * linesPerTask
	endLine = (rank * linesPerTask)	- 1
	sum = 0
	currLine = 0
	
	dataFile = open("test_data.txt", "r")
	for line in dataFile:
		if currLine >= startLine and currLine <= endLine:
			sum += int(line)
		currLine += 1	

	data = sum / linesPerTask
data = comm.gather(data, root = 0)

if rank == MASTER:
	globalAverage = float(sum(data[1:])) / (size - 1) # don't include first element
	print ("{0}".format(globalAverage))
