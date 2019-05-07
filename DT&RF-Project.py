
# coding: utf-8

# In[2]:


import pandas as pd, numpy as np


# In[3]:


import matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


loans = pd.read_csv('loan_data.csv')


# In[5]:


loans.head()


# In[6]:


loans.info()


# In[7]:


loans.describe()


# In[8]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[9]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Not fully paid = 1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Not fully paid = 0')
plt.legend()
plt.xlabel('FICO')


# In[10]:


plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# In[11]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# In[12]:


plt.figure(figsize=(11,7))
sns.lmplot(x='fico',y='int.rate',col='not.fully.paid',hue='credit.policy',data=loans,palette='Set1')


# In[13]:


loans.info()


# In[14]:


cat_feats = ['purpose']


# In[15]:


cat_feats


# In[16]:


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[18]:


final_data.info()


# In[57]:


from sklearn.model_selection import train_test_split


# In[63]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[64]:


from sklearn.tree import DecisionTreeClassifier


# In[65]:


dtree = DecisionTreeClassifier()


# In[66]:


dtree.fit(X_train,y_train)


# In[67]:


predictions = dtree.predict(X_test)


# In[68]:


from sklearn.metrics import classification_report,confusion_matrix


# In[71]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[72]:


from sklearn.ensemble import RandomForestClassifier


# In[84]:


rfc = RandomForestClassifier(n_estimators=300)


# In[85]:


rfc.fit(X_train,y_train)


# In[86]:


rfc_pred = rfc.predict(X_test)


# In[87]:


print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))

