import numpy as np

class Function:
  def __init__(self,f,df):
    self.f=f
    self.d=df

def sigmoid(x):
  return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
  return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
  return np.tanh(x)

def tanh_der(x):
  return 1.0 - x**2

def rand():
  return np.random.rand()

def setactivation(activation):
  if activation == 's':
    return Function(sigmoid,sigmoid_der)
  elif activation == 'tanh':
    return Function(tanh,tanh_der)