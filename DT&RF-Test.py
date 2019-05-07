
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('kyphosis.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[9]:


sns.pairplot(df,hue='Kyphosis')


# In[10]:


from sklearn.model_selection import train_test_split


# In[18]:


X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[19]:


from sklearn.tree import DecisionTreeClassifier


# In[20]:


dtree = DecisionTreeClassifier()


# In[21]:


dtree.fit(X_train,y_train)


# In[22]:


predictions = dtree.predict(X_test)


# In[23]:


from sklearn.metrics import classification_report,confusion_matrix


# In[24]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[25]:


from sklearn.ensemble import RandomForestClassifier


# In[26]:


rfc = RandomForestClassifier(n_estimators=200)


# In[27]:


rfc.fit(X_train,y_train)


# In[28]:


rfc_pred = rfc.predict(X_test)


# In[29]:


print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))


# In[31]:


df['Kyphosis'].value_counts()


# In[32]:


from IPython.display import Image


# In[33]:


from sklearn.externals.six import StringIO


# In[34]:


from sklearn.tree import export_graphviz


# In[35]:


import pydot


# In[36]:


features = list(df.columns[1:])
features


# In[39]:


dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())


# In[40]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)


# In[41]:


rfc_pred = rfc.predict(X_test)


# In[42]:


print(confusion_matrix(y_test,rfc_pred))

