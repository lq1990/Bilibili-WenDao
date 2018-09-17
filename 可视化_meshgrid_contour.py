__author__ = '文刀'

'visualization'

import numpy as np
import matplotlib.pyplot as plt

# meshgrid
x = np.arange(0,11,0.1)
y = np.arange(0,11,0.1)
xx,yy = np.meshgrid(x,y)
print('xx:',xx)
print('yy:',yy)
# contour, contourf

zz = np.sin(xx) + np.cos(yy)

plt.contourf(xx,yy,zz,3,cmap=plt.cm.rainbow)
# plt.contourf(xx,yy,zz,3)

cont=plt.contour(xx,yy,zz,3,colors='black')
plt.clabel(cont,inline=True,fontsizes=10)
plt.show()







