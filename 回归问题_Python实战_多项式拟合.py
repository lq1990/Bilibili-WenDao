# loaddataset
# regression


__author__ = '文刀'

# import numpy, matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt


# def loaddata
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

def scaling(mat): # 优化的地方
    mean = np.mean(mat,0)
    std = np.std(mat,0)
    new_mat = (mat-mean)/std
    return new_mat, mean, std


def wb_calc(xmat, ymat,lam=0,alpha=0.0001,maxIter=20000):
    m,n = xmat.shape

    np.random.seed(1)
    X = np.mat(np.zeros((m,m-1)))
    X[:,0] = xmat
    X[:,1] = xmat.A ** 2
    X[:,2] = xmat.A ** 3

    X, X_mean, X_std = scaling(X) # 优化的地方
    print('X_mean:',X_mean)
    print('X_std:',X_std)

    # init w b
    W = np.mat(np.random.randn(3,1))
    b = np.mat(np.random.randn(1,1))
    W0 = W.copy()
    b0 = b.copy()
    for i in range(maxIter):
        # dw, db
        H = X*W+b
        dw = 1/m * X.T*(H-ymat) + 1/m * lam*W
        #           (3,4)(4,1) + (3,1) = (3,1)
        db = 1/m * np.sum(H-ymat) #(1,1)

        # w,b update
        W -= alpha * dw
        b -= alpha * db
    return W,b,W0,b0,X_mean, X_std # return 多了几个


# show
xmat, ymat = loaddata('regression_data.txt')
print('xmat:',xmat,xmat.shape,type(xmat))
print('ymat:',ymat,ymat.shape,type(ymat))

W,b,W0,b0,X_mean, X_std = wb_calc(xmat,ymat,100,0.0001,50000) # 添加了一些

# plot
xrange = np.arange(1,7,0.001)             # 泛化

plotx1 = (xrange-X_mean[0,0])/X_std[0,0] # 优化的地方
plotx2 = (xrange**2-X_mean[0,1])/X_std[0,1]
plotx3 = (xrange**3-X_mean[0,2])/X_std[0,2]

w1 = W[0,0]
w2 = W[1,0]
w3 = W[2,0]

ploth = w1*plotx1 + w2*plotx2 + w3*plotx3 +b[0,0]  # end
plt.plot(xrange,ploth,label='h_end')

w1_0 = W0[0,0]
w2_0 = W0[1,0]
w3_0 = W0[2,0]

ploth0 = w1_0*plotx1 + w2_0*plotx2 + w3_0*plotx3 + b0[0,0] # init
# plt.plot(xrange,ploth0,label='h_init')

plt.scatter(xmat.A,ymat.A,s=50,c='r',label='dataset')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()



