import numpy as np
from ptkNN import *

inp=np.random.rand(3,4)

n1=Network(5,[3,5,2,4,1],[3,5,2,4,1],["s","tanh","s","s","s"])
n1.memorize()

n1.log.read(1)