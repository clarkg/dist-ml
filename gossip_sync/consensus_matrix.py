import numpy as np

def generate_consensus_matrix(A):
    """ 
    Given an adjacency matrix A,
    return a doubly stochastic consensus matrix P
    computed using the Metropolis-Hastings algorithm

    A : square adjacency matrix

    output : consensus matrix of same size as A

    """

    size_A = A.shape[0]

    # size "size_A" identity matrix
    I = np.eye(size_A, dtype=np.float)
    A = A + I

    # iterate through vertices
    # and get their degree
    dv = np.zeros(size_A, dtype=np.float)
    for v in range(0,size_A):
        for i in range(0,size_A):
            dv[v] += A[v][i]

    # calculate non diagonal Pij
    P = np.zeros([size_A, size_A])
    for i in range(0,size_A):
        for j in range(0,size_A):
            if i != j and A[i][j] != 0:
                P[i][j] = 1/max(dv[i],dv[j])

    # calculate Pii
    for i in range(0,size_A):
        sum_Pij = 0
        for j in range(0,size_A):           
            sum_Pij += P[i][j]
        P[i][i] = 1 - sum_Pij

    return P

def usr_def_adjacency_matrix():
    # Paste your adjacency matrix here!
    return None

def adjacency_matrix(size):
    if size == 1:
        return np.array([[1]])
    if size == 2:
        return np.array([
            [0, 1],
            [1, 0]
            ])
    if size == 4:
        return np.array([
          [0,1,0,1],
          [1,0,1,0],
          [0,1,0,1],
          [1,0,1,0]
          ]) 

    if size == 8:
        return np.array([
            [0,1,0,0,0,0,0,1],
            [1,0,1,0,0,0,0,0],
            [0,1,0,1,0,0,0,0],
            [0,0,1,0,1,0,0,0],
            [0,0,0,1,0,1,0,0],
            [0,0,0,0,1,0,1,0],
            [0,0,0,0,0,1,0,1],
            [1,0,0,0,0,0,1,0]
            ])
