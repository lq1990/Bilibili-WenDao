# 神经网络 划分 太极图

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

def scaling(data):
    min = np.min(data,0)
    max = np.max(data,0)
    new_data = (data-min)/(max-min)
    return new_data,min,max

def sigmoid(data):
    return 1/(1+np.exp(-data))

def calc_wb(X,label,n_hidden_dim=10,alpha=0.05,reg_lambda=0,num_step=10000,anim_inter=100):
    # init. w b
    W1 = np.mat(np.random.randn(2,n_hidden_dim)) # (2,3)
    b1 = np.mat(np.random.randn(1,n_hidden_dim)) # (1,3)
    W2 = np.mat(np.random.randn(n_hidden_dim,1)) # (3,1)
    b2 = np.mat(np.random.randn(1,1)) # (1,1)
    w1_save = []
    b1_save = []
    w2_save = []
    b2_save = []
    step = []
    for stepi in range(num_step):
        # FP
        z1 = X*W1 + b1 # (20,2)(2,3)+(1,3)=(20,3)
        a1 = sigmoid(z1) # (20,3)
        z2 = a1*W2 + b2 # (20,3)(3,1)+(1,1)=(20,1)
        a2 = sigmoid(z2) # (20,1)
        # BP
        delta2 = a2-label # (20,1)
        dW2 = a1.T*delta2 + reg_lambda*W2
        #    (3,20)(20,1) = (3,1)
        # db2 = np.sum(delta2,0)
        db2 = np.mat(np.ones((X.shape[0],1))).T * delta2

        delta1 = np.mat((delta2*W2.T).A*a1.A*(1-a1).A) # (20,3)
        dW1 = X.T*delta1 + reg_lambda*W1
        db1 = np.sum(delta1,0)
        # new W,b
        W2 -= alpha*dW2
        b2 -= alpha*db2
        W1 -= alpha*dW1
        b1 -= alpha*db1
        if stepi % anim_inter == 0:
            w1_save.append(W1.copy())
            b1_save.append(b1.copy())
            w2_save.append(W2.copy())
            b2_save.append(b2.copy())
            step.append(stepi)

    return W1,b1,W2,b2,w1_save,b1_save,w2_save,b2_save,step

# implement
xmat, ymat = loaddata('nn_data_tj.txt')
xmat_s,xmat_min,xmat_max = scaling(xmat)
# print('x:',xmat,xmat.shape,type(xmat))
# print('y:',ymat,ymat.shape,type(ymat))
# s_xmat = scaling(xmat)
# print('s_x:',s_xmat)

anim = 0  #  0: no anim. 1: yes

if anim == 0:
    W1,b1,W2,b2,w1_save,b1_save,w2_save,b2_save,step=calc_wb(xmat_s,ymat,n_hidden_dim=10,alpha=0.05,reg_lambda=0,num_step=30000)
if anim == 1:
    W1,b1,W2,b2,w1_save,b1_save,w2_save,b2_save,step=calc_wb(xmat_s,ymat,n_hidden_dim=10,alpha=0.05,reg_lambda=0,num_step=10000,anim_inter=100)

# contour
xx1=np.arange(-0.5,10.5,0.01)
xx2=np.arange(-0.5,10.5,0.01)
XX1,XX2=np.meshgrid(xx1,xx2)
plotx_old = np.c_[XX1.ravel(),XX2.ravel()]
plotx = (plotx_old-xmat_min)/(xmat_max-xmat_min)


# animation
if anim == 1:
    for i in range(len(w1_save)):
        plt.clf()
        plotz1 = plotx*w1_save[i]+b1_save[i]
        plota1 = sigmoid(plotz1)
        plotz2 = plota1*w2_save[i] + b2_save[i]
        plota2 = sigmoid(plotz2)
        ploty = np.reshape(plota2,XX1.shape)
        plt.contourf(XX1,XX2,ploty,3,alpha=0.5,cmap=plt.cm.rainbow)
        cont=plt.contour(XX1,XX2,ploty,3)
        plt.clabel(cont,inline=True,fontsize=10)
        plt.scatter(xmat[:,0][ymat==0].A,xmat[:,1][ymat==0].A,s=50,marker='o',label='0',cmap=plt.cm.rainbow)
        plt.scatter(xmat[:,0][ymat==1].A,xmat[:,1][ymat==1].A,s=50,marker='o',label='1',cmap=plt.cm.rainbow)
        plt.grid()
        plt.legend(loc=1)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.title('神经网络划分太极图 step:%s'%np.str(step[i]))
        plt.pause(0.0001)
    plt.show()
if anim == 0:
    plotz1 = plotx * W1 + b1
    plota1 = sigmoid(plotz1)
    plotz2 = plota1 * W2 + b2
    plota2 = sigmoid(plotz2)
    ploty = np.reshape(plota2, XX1.shape)
    plt.contourf(XX1, XX2, ploty, 1, alpha=0.5,cmap=plt.cm.rainbow)
    cont = plt.contour(XX1, XX2, ploty, 1)
    plt.clabel(cont, inline=True, fontsize=10)
    plt.scatter(xmat[:, 0][ymat == 0].A, xmat[:, 1][ymat == 0].A, s=50,marker='o', label='0',cmap=plt.cm.rainbow)
    plt.scatter(xmat[:, 0][ymat == 1].A, xmat[:, 1][ymat == 1].A, s=50,marker='o', label='1',cmap=plt.cm.rainbow)
    plt.grid()
    plt.legend(loc=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('神经网络划分太极图')
    plt.show()






