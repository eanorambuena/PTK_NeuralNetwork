import numpy as np

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

class LogFile:
  def __init__(self):
    self.text=""
  def row(self,t):
    self.text=self.text+str(t)+","
  def section(self):
    self.text=self.text+"|"
  def export(self):
    f=open("log.txt","w")
    f.write(self.text)
    f.close()
  def read(self,printYesOrNo=0):
    f=open("log.txt","r")
    k=f.read()
    self.text=k
    f.close()
    if(printYesOrNo):
      t=""
      for i in range(0,len(k)):
        if k[i]==',':
          print(t)
          t=""
        if k[i]=='|':
          print("")
        elif k[i]=='.' or k[i].isalnum():
          t+=k[i]
  
class Function:
  def __init__(self,f,df):
    self.f=f
    self.d=df

class Neuron:
  def __init__(self, size, activation='s'):
    self.size=size
    self.lastcompute=np.pi
    self.setactf(activation)
    self.reset()
  def reset(self):
    self.b=rand()
    self.w=[]
    for i in range(0,self.size):
      self.w.append(rand())
  def setactf(self,activation):
    if activation == 's':
      self.a=Function(sigmoid,sigmoid_der)
    elif activation == 'tanh':
      self.a=Function(tanh,tanh_der)
  def memorize(self):
    self.log=LogFile()
    self.log.row("B")
    self.log.row(self.b)
    self.log.row("W")
    for i in range(0,self.size):
      self.log.row(self.w[i])
  def compute(self,v):
    try:
      self.lastcompute=self.a.f(self.b+np.dot(np.array(self.w),np.array(v)))
    except:
      self.log.row("No input given. Last compute: "+str(self.lastcompute))

class Layer:
  def __init__(self, size, inputSize, activation='s'):
    self.a=activation
    self.size=size
    self.insize=inputSize
    self.lastcompute=np.random.rand(size)
    self.reset()
  def reset(self):
    self.l=[]
    for i in range(0,self.size):
      self.l.append(Neuron(self.insize,self.a))
  def memorize(self):
    self.log=LogFile()
    for i in range(0,self.size):
      self.log.row("N "+str(i))
      self.l[i].memorize()
      self.log.row(self.l[i].log.text)
      self.log.section()
  def compute(self,inp):
    for i in range(0,self.size):
      self.l[i].compute(inp[i])
      self.lastcompute[i]=self.l[i].lastcompute

class Network:
  def __init__(self, depth, sizev, inputSizev, activationv):
    self.depth=depth
    self.av=activationv
    self.sizev=sizev
    self.insizev=inputSizev
    self.reset()
  def reset(self):
    self.layers=[]
    for i in range(0,self.depth):
      self.layers.append(Layer(self.sizev[i],self.insizev[i],self.av[i]))
  def memorize(self):
    self.log=LogFile()
    for i in range(0,self.depth):
      self.log.row("L "+str(i))
      self.layers[i].memorize()
      self.log.row(self.layers[i].log.text)
      self.log.section()