#!/usr/bin/env python
# coding: utf-8

# # LetsGrow More: Data Science Internship
# ## Task 2 : (Intermediate  Level Task) Exploratory Data Analysis of Terrorism Dataset.
# ### Intern name : Gadade Kailas Rayappa
# ### Step 1 : Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Step 2 : Loading the dataset

# In[2]:


data = pd.read_csv("C:\\Users\\Kailas\\OneDrive\\Desktop\\globalterrorismdb_0718dist.csv", encoding = "Latin", low_memory=False)
data.head()


# In[3]:


data=data[["iyear","imonth","iday","country_txt","region_txt","provstate","city",
           "latitude","longitude","location","summary","attacktype1_txt","targtype1_txt",
           "gname","motive","weaptype1_txt","nkill","nwound","addnotes"]]
data.head()


# In[4]:


data.rename(columns={"iyear":"Year","imonth":"Month","iday":"Day","country_txt":"Country",
                    "region_txt":"Region","provstate":"Province/State","city":"City",
                    "latitude":"Latitude","longitude":"Longitude","location":"Location",
                    "summary":"Summary","attacktype1_txt":"Attack Type","targtype1_txt":"Target Type",
                    "gname":"Group Name","motive":"Motive","weaptype1_txt":"Weapon Type",
                    "nkill":"Killed","nwound":"Wounded","addnotes":"Add Notes"},inplace=True)
data.head()


# In[5]:


data.info()


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# In[8]:


data["Killed"]=data["Killed"].fillna(0)
data["Wounded"]=data["Wounded"].fillna(0)
data["Casualty"]=data["Killed"]+data["Wounded"]
data.describe()


# #### The data consists information from Year 1970 to 2017.
# #### The maximum peoples killed in attack weree 1570.

# ### Step 3 : Performing EDA
# ### Regionwise Analysis

# In[9]:


plt.subplots(figsize=(15,6))
sns.countplot('Year',data=data,palette='spring_r',edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# ### here The graph shows the number of Attacks in each year from 1970 to 1992 and here we observed that the number of global terrorism attacks are continuously increasing from 1988. The Graph shows 1992 is the most unlucky year because the most number of atacks took place in the year 1992.

# In[10]:


data.Year.plot(kind = 'hist', color = 'pink', bins=range(1970, 2018), figsize = (16,7),grid=True)
plt.xticks(range(1970, 2018), rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Years", fontsize=15)
plt.ylabel("Number of Attacks", fontsize=15)
plt.title("Number of Attacks By Year", fontsize=16)
plt.show()


# In[11]:


plt.figure(figsize=(15,6))
sns.countplot(data.Region, order = data['Region'].value_counts().index)
plt.xlabel("Region")
plt.xticks(rotation = 90)
plt.ylabel("Number of Assassinations")
plt.title("Assassinations by Region")
plt.show()


# In[12]:


killed=data["Region"].value_counts(dropna=False).sort_index().to_frame().reset_index().rename(columns={"index":"Region","Region":"Killed"}).set_index("Region")
killed.head()


# In[13]:


killed.plot(kind="bar",color="pink",figsize=(15,6),fontsize=15)
plt.title('Terrorist attack',fontsize=12)
plt.xlabel('Region',fontsize=12)
plt.ylabel('No of peoples killed',fontsize=12)
plt.show()


# In[14]:


data['Group Name'].value_counts().to_frame().drop("Unknown").head(10).plot(kind="bar",color="pink", figsize=(20,10)) 
plt.title("Top 10 terrorist group attack", fontsize=25)
plt.xlabel("terrorist group name", fontsize=20) 
plt.ylabel("Attack number", fontsize=20)
plt.show()


# ### Here the graph shows the top 10 terrorist group attacks and Taliban is the most active terrorist group followed by the others.

# In[15]:


plt.figure(figsize=(25,12))
sns.countplot(data["Target Type"],order=data["Target Type"].value_counts().index,edgecolor='k')
plt.xlabel('Target Type',fontweight='bold',fontsize=18)
plt.ylabel('Number Targets',fontweight='bold',fontsize=18)
plt.title('Type of Targets',fontweight='bold',fontsize=25)


# ### The main attack target is Private citizens and proverty.

# In[16]:


plt.figure(figsize=(25,12))
sns.barplot(data["Province/State"].value_counts()[:10].index,data["Province/State"].value_counts()[:10].values,edgecolor='k')
plt.xlabel('States',fontweight='bold',fontsize=18)
plt.ylabel('Counts',fontweight='bold',fontsize=18)
plt.title('Top 10 most affected terriost attack States',fontweight='bold',fontsize=25)
plt.xticks(rotation=60)
plt.show()


# ### The most affected terriost attack State is Baghdad.

# In[17]:


pd.crosstab(data.Year, data.Region).plot(kind='area',figsize=(15,6))
plt.title('Terrorist Activities by Region in each Year')
plt.ylabel('Number of Attacks')
plt.show()


# ### Thank You...

# In[ ]:




