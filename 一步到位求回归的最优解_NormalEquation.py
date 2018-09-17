# Normal Equation vs Gradient descent

__author__ = '文刀'

import numpy as np
import matplotlib.pyplot as plt

# loaddata def
def loaddata(filename):
    fr = open(filename)
    x = []
    y = []
    for line in fr.readlines():
        line = line.strip().split()
        x.append(float(line[0]))
        y.append(float(line[1]))
    xmat = np.mat(x).T
    ymat = np.mat(y).T
    return xmat, ymat

# normal equation def
def normalequation(xmat,ymat,lam=0):
    X = np.mat(np.zeros((4,4)))
    X[:,0] = 1
    X[:,1] = xmat
    X[:,2] = xmat.A**2
    X[:,3] = xmat.A**3

    eye = np.mat(np.eye(4))
    eye[0,0] = 0
    print('eye:\n',eye)
    W = (X.T*X-lam*eye).I *X.T * ymat
    return W


# implement
xmat,ymat = loaddata('ytb_03.txt')
print('xmat:',xmat)
print('ymat:',ymat)

lam=0.001
w0 = normalequation(xmat,ymat,0)
w1 = normalequation(xmat,ymat,lam)
print('w:',w0)
# plot
plotx = np.arange(1.9,6.1,0.01)
ploth0 = w0[0,0] + w0[1,0]*plotx + w0[2,0]*plotx**2 + w0[3,0]*plotx**3
ploth1 = w1[0,0] + w1[1,0]*plotx + w1[2,0]*plotx**2 + w1[3,0]*plotx**3
plt.plot(plotx,ploth0,label='from normal equation, lam=0')
plt.plot(plotx,ploth1,label='from normal equation, lam='+np.str(lam))

plt.scatter(xmat.A,ymat.A,c='r',s=100,label='dataset')
plt.grid()
plt.legend()
plt.show()




