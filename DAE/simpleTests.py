import numpy

A = numpy.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mean = A.mean(axis=0)
print A - mean[numpy.newaxis,:]
