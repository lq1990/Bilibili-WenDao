# 内插法

__author__ = '文刀'

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def loaddata(filename):
    fr = open(filename)
    xlist = []
    ylist = []
    for line in fr.readlines():
        line = line.strip().split()
        xlist.append(float(line[0]))
        ylist.append(float(line[-1]))
    xmat = np.mat(xlist).T
    ymat = np.mat(ylist).T
    return xmat, ymat

# interpolation
def interpolation(xmat, ymat):
    X = np.mat(np.zeros((4,4)))
    X[:,0] = xmat
    X[:,1] = xmat.A **2
    X[:,2] = xmat.A **3
    X[:,3] = 1

    W = X.I * ymat
    return W





# implement
xmat, ymat = loaddata('interpolation.txt')
print('xmat:',xmat)
print('ymat:',ymat)

plt.scatter(xmat.A,ymat.A,s=50,c='r',label='dataset')

w = interpolation(xmat,ymat)
plotx = np.arange(1.9,6.1,0.001)
ploty = plotx*w[0,0] + plotx**2*w[1,0] + plotx**3 * w[2,0] + 1*w[3,0]
plt.plot(plotx,ploty,label='interpolation')

plt.grid(True)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()



