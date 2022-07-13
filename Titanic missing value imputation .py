#!/usr/bin/env python
# coding: utf-8

# Overview
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
# 
# Variable Notes
# 
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# 
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.
# 
# Survived(o/v)= 0:No,1:Yes.

# In[1]:


#Standard libaries for data analysis.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Using pandas setoption method to see all the columns in dataset.
pd.set_option('display.max_columns', None)


# In[6]:


# Loadingthe file.

train=pd.read_csv("D:/Python/Titanic Dataset/train.csv")


# In[7]:


# Checking shape of the data.

print(train.shape)


# ##### Working on imputing the missing values by developing an ML model to predict the missing values in the dataset

# In[8]:


#checking for null values.

train.isnull().sum()


# In[9]:


# % of data missing each column.
for i in train.columns:
    percent=round((train[i].isnull().mean()*100),2)
    print(f"{i}: {percent}%.")


# -- Age has around 20% missing values.
# 
# -- Embarked has has only 2 missing values.
# 
# -- Cabin has 77% missing values.
# 
# Hence it would suffice to drop the Cabin column  and null value point from empbarked as this will be relevant in predition model.

# In[10]:


df=train.drop(['Cabin'],axis=1)


# In[11]:


df


# Now we are left with Age and Embarked which are having null values.

# In[12]:


# Working with embarked.
df=df.dropna(subset=['Embarked'],axis=0,inplace=False)


# In[13]:


df


# In[14]:


df.isnull().sum()


# ##### Visualising the data to see pattern or relations so that null values in age column can be imputed.

# In[15]:


#correlation

cor=df.corr()


# In[16]:


cor


# In[17]:


plt.figure(figsize=(8,6))
sns.heatmap(cor,annot=True)

Observation:
1. Survived is  positively correlated with Parch and Fare.
2. parch and fare and sibsp has positive correlation with the output variable.
3. pclass is positively correlated with sibsip.
4. age is positively correlated with fare.
# ##### Comparing Age with Pclass and Sex because these feature might help imputing the missing values in age column.

# In[18]:


sns.boxplot(x='Pclass',y='Age',data=df)


# Basis age we can see that mostly Pclass1 had aged people, whereas class 2 and 3 people were mostly young.

# In[19]:


sns.boxplot(x='Sex',y='Age',data=df)


# The average age of male and female travelling are almost similar which is 29,27 respectively.

# In[20]:


sns.countplot(x='Pclass',hue='Sex',data=df)
plt.legend()


# In[21]:


sns.countplot(df.Sex)


# In[22]:


sns.countplot(df.Pclass)


# ##### After we got the mean ages of people travelling basis class they were travelling and sex. Also we can se that max no. people travelling are males and simultaneously max no. of people travelling are from pclass 3. So filling the age nan values basis these columns can be done, however there would a possibilty of gettiing some extreme value and the result can be baised. Hence developing a model using random forest as it is not affected by outliers to predit age.

# In[23]:


# Dropping columns that won't help in our prediction.

df=df.drop(['Ticket','Name'],axis=1)


# In[24]:


df


# ## Developing a model to predict Age.

# ###### Processing the dataset as we cannot apply model on categorical feature

# In[25]:


#Using onehotencoder changing the sex and embarked data into numerical form.

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder=ColumnTransformer([('one',OneHotEncoder(),[3,8])],remainder='passthrough')


# In[26]:


z=encoder.fit_transform(df)


# In[27]:


z


# In[28]:


#Converting the array to dataframe using pandas.
z=pd.DataFrame(z)


# In[29]:


z


# In[30]:


#Renamimg the columns.
z=z.rename(columns={
    0:'Female',
    1:'Male',
    2:'C',
    3:'Q',
    4:'S',
    5:'PassengerId',
    6:'Survived',
    7:'Pclass',
    8:'Age',
    9:'Sibsp',
    10:'Parch',
    11:'Fare'
})


# In[31]:


z


# #### Developing the model to predict age(missing values)

# Step 1: Separate the null values from the data frame (z) and create a variable “test data”.

# In[32]:


df_test=z[z['Age'].isnull()]
df_test


# Step 2: Drop the null values from the data frame (z) and represent them as ‘train data”

# In[33]:


z.dropna(axis=0,inplace=True)


# In[34]:


z


# Step 3: Create “x_train” & “y_train” from train data.

# In[35]:


xtrain=z.drop(['Age'],axis=1)
xtrain


# In[36]:


ytrain=z.Age

ytrain


# ##### Building a linear regression model 

# In[37]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[38]:


model.fit(xtrain,ytrain)


# Step 5: Create the x_test from test data

# In[39]:


xtest=df_test[["Female","Male","C","Q","S","PassengerId","Survived","Pclass","Sibsp","Parch","Fare"]]


# In[40]:


xtest


# Step 6: Apply the model on x_test of test data to make predictions. here, we have created a new variable ‘y_pred’.

# In[41]:


ypred=model.predict(xtest)

ypred


# As above we can see that using linear regression we got the missing values but there are some negative value in our prediction and as age cannot be neagtive it would be suffice to build another model to predict age.

# #### Building Random forest model 

# In[42]:


from sklearn.ensemble import RandomForestRegressor


# In[43]:


model1=RandomForestRegressor()
model1.fit(xtrain,ytrain)


# In[44]:


ypred1=model.predict(xtest)

ypred1


# As there are still some negative values let's try to develop another model.

# In[45]:


get_ipython().system('pip install xgboost')


# In[46]:


#Now using xgboost for our prediction.
import xgboost


# In[47]:


xgb=xgboost.XGBRegressor()


# In[48]:


xgb.fit(xtrain,ytrain)


# In[49]:


ypred2=xgb.predict(xtest)

ypred2


# Finally the age has has been predicted without negative value and missing values

# In[50]:


len(ypred2)


# Now we got the all the age values and adding predicted values to the original dataframe for futher analysis.

# ### Splitting the dataframe basis null and non-values.

# In[51]:


# df is dataframe were all the age values are null
df2=df[df['Age'].isnull()]
df2


# In[52]:


#Dropping all the null values fron the original dataframe.
df.dropna(axis=0,inplace=True)
df


# We now have two seperate dataframe splitted basis null and non-null values in age column.

# In[53]:


# Replacing all the null vales with the predicted values.
df2.Age=ypred2


# In[54]:


df2


# In[55]:


df2.isnull().sum()


# In[56]:


# After subsitituting the null values now mergeing both the dataframe.
df=pd.concat([df,df2],axis=0)


# In[57]:


df


# In[58]:


#Checking null values.
df.isnull().sum()


# ##### Now, we do not have any null values, so let's analyze the data futher.

# In[61]:


df.to_csv(r'D:/Python/Titanic Dataset/titanic_clean.csv')

