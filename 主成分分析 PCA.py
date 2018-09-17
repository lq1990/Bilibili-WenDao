
# coding: utf-8

# In[1]:


# 文刀出品
# 代码实现 PCA
# 使用数据来源于 kaggle Titanic
# 使用 pandas 对数据进行预处理（非本期重点）
# 本视频使用到的 dataset 会上传到我 GitHub 
# https://github.com/lq1990/Bilibili-WenDao.git
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


traindata_0 = pd.read_csv('train_titanic.csv',index_col='PassengerId')
traindata_0.info()
traindata_0.head()
# 对缺失数据的声明，Age，Embarked，Cabin
# 先使用常规方法处理缺失数据。本期重点在PCA


# In[3]:


traindata = traindata_0.copy()
# replace
traindata.Sex = traindata.Sex.replace({'female':0,'male':1}) # one hot
traindata.Embarked = traindata.Embarked.replace({'C':0,'Q':1,'S':2})
# fillna
traindata.Age.fillna(traindata.Age.mean(),inplace=True)
traindata.Embarked.fillna(method='ffill',inplace=True)
# drop
traindata.drop(columns=['Survived','Name','Ticket','Cabin'],inplace=True)
traindata.info()
print(traindata.head())

traindata.to_csv('data_pca.csv') 
# 此csv文件会上传到我 Github账号，可自行下载。


# In[28]:


# PCA期视频重点从这里开始
# read data
data = pd.read_csv('data_pca.csv',index_col='PassengerId')
data.describe()
# normalization
def norm_(x):
    xmean = np.mean(x,0)
    std = np.std(x,0)
    return (x-xmean)/std
data_ = norm_(data)
data_.describe()

# V
# ew, ev = np.linalg.eig(data_.T.dot(data_))
ew, ev = np.linalg.eig(np.cov(data_.T))
ew_order = np.argsort(ew)[::-1]
ew_sort = ew[ew_order]
ev_sort = ev[:,ew_order]
print(ew_sort)
print(ev_sort)
pd.DataFrame(ew_sort).plot(kind='bar')
# V
V = ev_sort[:,:2]
# Xnew
X_new = data_.dot(V)
# scatter
get_ipython().run_line_magic('matplotlib', 'notebook')
sc = plt.scatter(X_new.iloc[:,0],X_new.iloc[:,1],s=5,c=traindata_0.Survived,cmap=plt.cm.coolwarm)
plt.xlabel('PC 0')
plt.ylabel('PC 1')
plt.colorbar(sc)

print(V)
print(data_.columns)


# In[ ]:




