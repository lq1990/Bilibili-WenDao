# neural networks

__author__ = '文刀'

#
import numpy as np
import matplotlib.pyplot as plt

# loaddata
def loaddata(filename):
    fr = open(filename)
    x = []
    y = []
    for line in fr.readlines():
        line = line.strip().split()
        x.append([float(line[0]),float(line[1])])
        y.append([float(line[-1])])
    return np.mat(x), np.mat(y)

# data scaling
def scaling(data):
    max = np.max(data,0)
    min = np.min(data,0)
    return (data-min)/(max-min),max,min

# sigmoid
def sigmoid(data):
    return 1/(1+np.exp(-data))

# w b calc
def wb_calc(X,ymat,alpha=0.1,maxIter=10000,n_hidden_dim=3,reg_lambda=0):
    # init w b
    W1 = np.mat(np.random.randn(2,n_hidden_dim))
    b1 = np.mat(np.random.randn(1,n_hidden_dim))
    W2 = np.mat(np.random.randn(n_hidden_dim, 1))
    b2 = np.mat(np.random.randn(1, 1))
    w1_save = []
    b1_save = []
    w2_save = []
    b2_save = []
    ss = []
    for stepi in range(maxIter):
        # FP
        z1 = X*W1 + b1 # (20,2)(2,3) + (1,3) = (20,3)
        a1 = sigmoid(z1) # (20,3)
        z2 = a1*W2 + b2 # (20,3)(3,1) + (1,1) = (20,1)
        a2 = sigmoid(z2) # (20,1)
        # BP
        a0= X.copy()
        delta2 = a2 - ymat # (20,1)
        delta1 = np.mat((delta2*W2.T).A * (a1.A*(1-a1).A))
        #              (20,1)(1,3) .* (20,3) = (20,3)
        dW1 = a0.T*delta1 + reg_lambda*W1 # (2,20)(20,3) + (2,3) = (2,3)
        db1 = np.sum(delta1,0)
        # db1 = np.mat(np.ones()) * delta1
        dW2 = a1.T*delta2 + reg_lambda*W2
        db2 = np.sum(delta2,0)
        # undate w b
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        if stepi % 100 ==0:
            w1_save.append(W1.copy())
            b1_save.append(b1.copy())
            w2_save.append(W2.copy())
            b2_save.append(b2.copy())
            ss.append(stepi)
    return W1,b1,W2,b2,w1_save,b1_save,w2_save,b2_save,ss


# implement
xmat, ymat = loaddata('nn_data_2.txt')
# print('x:',xmat,xmat.shape,type(xmat))
# print('y:',ymat,ymat.shape,type(ymat))
xmat_s,xmat_max,xmat_min = scaling(xmat)

W1,b1,W2,b2,w1_save,b1_save,w2_save,b2_save,ss=wb_calc(xmat_s,ymat,0.05,20000,10,0)

# show

plotx1 = np.arange(0,10,0.01) # array
plotx2 = np.arange(0,10,0.01)
plotX1,plotX2=np.meshgrid(plotx1,plotx2)
plotx_new = np.c_[plotX1.ravel(),plotX2.ravel()]
# 上面命令中，做 c_ 的意义在于将方阵转换成一列数，两个方阵转换成两列
# 也就是将 meshgrid 后的所有网格点的横、纵坐标（分别对应2个feature）
# 由方形变成两个列向量。
# 因为我们NN的输入是两列数
# 将网格点的坐标x、y由方形变成两列矩阵后，
# 也就是泛化求解网格点上所有点，作为NN输入求解NN模型的output
# 因为NN输出的维度也是列向量，一列数。在contourf之前还需要把它reshape
# 来对应方形网格上每一个点的 高度。最终就把网格上所有点的等高图画出来了
plotx_new2 = (plotx_new-xmat_min)/(xmat_max-xmat_min) # testdata scaling

for i in range(len(w1_save)):
    plt.clf()
    plot_z1 = plotx_new2*w1_save[i] + b1_save[i]
    plot_a1 = sigmoid(plot_z1)
    plot_z2 = plot_a1*w2_save[i] + b2_save[i]
    plot_a2 = sigmoid(plot_z2)
    ploty_new = np.reshape(plot_a2,plotX1.shape)

    plt.contourf(plotX1,plotX2,ploty_new,1,alpha=0.5)

    plt.scatter(xmat[:,0][ymat==0].A,xmat[:,1][ymat==0].A,s=100,marker='o',label='0')
    plt.scatter(xmat[:,0][ymat==1].A,xmat[:,1][ymat==1].A,s=150,marker='^',label='1')
    plt.grid()
    plt.legend()
    plt.title('iter:%s'%np.str(ss[i]))
    plt.pause(0.001)
plt.show()




