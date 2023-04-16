#!/usr/bin/env python
# coding: utf-8

# # LetsGrow More: Data Science Internship
# ## Task 2 : (Intermediate Level Task) Prediction Using Decision Tree Algorithm.¶
# ### Intern name : Gadade Kailas Rayappa
# ### Step 1 : Importing Libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ### Step 2 : Importing Dataset

# In[2]:


data=pd.read_excel("C:\\Users\\Kailas\\OneDrive\\Desktop\\Data II.xlsx")
data


# In[3]:


data.head()


# In[4]:


data.tail()


# ### Step 2: Exploratory Data Analysis

# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


# unique value in each columns
for i in data.columns:
    print(i, "\t\t",len(data[i].unique()))


# In[8]:


list_columns=data.columns
list_columns


# In[9]:


data.describe()


# In[10]:


data.isnull().sum()


# In[11]:


data.isnull().values.any()


# In[12]:


features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
x= data.loc[:, features].values
print(x)


# In[13]:


y=data.Species
print(y)


# ### Step 3: Data Visualization comparing various features

# In[14]:


sns.pairplot(data)


# In[15]:


import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


plt.figure(figsize=(12,8))
sns.heatmap(data.describe(),annot=True,fmt='.2f',cmap='rainbow')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("visualize average,number0f,min,max,std,Queartile",fontsize=16)


# In[17]:


# Relationship between the data
data.corr()


# In[18]:


plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("visualize average,number0f,min,max,std,Queartile",fontsize=18)


# ### Step 4: Decision Tree Tree Model Training

# In[19]:


# Model Training
x_train, x_test, y_train, y_test= train_test_split(x, y, random_state=0)
clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
clf.fit(x_train, y_train)
clf.predict(x_test[0:1])


# In[20]:


from sklearn import preprocessing
data['Specices']=preprocessing.LabelEncoder().fit_transform(data['Species'])


# In[21]:


data.dtypes


# In[22]:


X=data.iloc[:,1:5].values
Y=data.iloc[:,-1].values


# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


# In[24]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[25]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)
print("The Decision Tree Classification model trained")


# In[26]:


classifier_tree=tree.DecisionTreeClassifier()
classifier_tree=classifier_tree.fit(X_train,Y_train)


# In[27]:


# text graph representation
text_presentation=tree.export_text(classifier_tree)
print(text_presentation)


# In[28]:


plt.figure(figsize=(18,16))
tree.plot_tree(classifier_tree,filled=True,impurity=True)


# ### Step 5: Calculating the Model accuracy

# In[29]:


y_pred=classifier.predict(X_test)
y_pred


# In[30]:


print("Accuracy score:",np.mean(y_pred==y_test))


# ### Making the confusion matrix

# In[31]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[32]:


score=np.mean(y_pred==y_test)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True,fmt='.0f',linewidths=0.5,square= True,cmap = 'ocean');
plt.ylabel('Actual label\n',fontsize = 14);
plt.xlabel('Predicted label\n', fontsize = 14);
plt.title('Accuracy Score: {}\n'.format(score),size =16);
plt.tick_params(labelsize= 15)
plt.show()


# In[33]:


cm_accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of model:",cm_accuracy)


# ### Classification_report

# In[34]:


from sklearn import metrics
print(metrics.classification_report(y_test, classifier.predict(X_test)))


# ### Conclusion
# #### The Classifier model can predict the Species of the flower 96% Accuracy score.

# In[ ]:




