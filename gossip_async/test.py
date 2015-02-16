import numpy as np

def stepGradient(m_current, b_current, points):
	b_gradient = 0
    	m_gradient = 0
    	N = float(len(points))
	for i in range(0, len(points)):
						
        	b_gradient += -(2./N) * (points[i][1] - ((m_current*points[i][0]) + b_current))
        	m_gradient += -(2./N) * points[i][0] * (points[i][1] - ((m_current * points[i][0]) + b_current))
	return b_gradient, m_gradient

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

w = np.array([0.0, 0.0])
training_data = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]

result = gradient(w, training_data)
m, b = stepGradient(0.0, 0.0, training_data)
print result
print m
print b
