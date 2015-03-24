import numpy as np
import sys

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

def isSquare(P):
    """Returns true if array P is a square array"""
    return P.ndim == 2 and P.shape[0] == P.shape[1]
