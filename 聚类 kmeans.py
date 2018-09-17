
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_table('kmeans_data.txt',header=None,names=['x','y'])
x = data['x']
y = data['y']

plt.scatter(x,y)


# In[3]:


data.head()


# In[5]:


def distance(data,centers):
#     data: 80x2, centers: 4x2
    dist = np.zeros((data.shape[0],centers.shape[0]))
    for i in range(len(data)):
        for j in range(len(centers)):
            dist[i,j] = np.sqrt(np.sum((data.iloc[i,:]-centers[j])**2))
        
    return dist
    

def near_center(data,centers):
    dist = distance(data,centers)
    near_cen = np.argmin(dist,1)
    return near_cen
    


def kmeans(data,k):
    # step 1: init. centers
    centers = np.random.choice(np.arange(-5,5,0.1),(k,2)) 
    print(centers)
    
    for _ in range(10):
        # step 2: 点归属
        near_cen = near_center(data,centers)
        # step 3：簇重心更新
        for ci in range(k):
            centers[ci] = data[near_cen==ci].mean()
    
    return centers,near_cen
    
    
centers,near_cen = kmeans(data,4)

plt.scatter(x,y,c=near_cen)
plt.scatter(centers[:,0],centers[:,1],marker='*',s=500,c='r')

