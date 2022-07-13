#!/usr/bin/env python
# coding: utf-8

# # Titanic dataset

# In the the dataset missing values has been imputed. Now, using the clean dataset to analyse it further.

# In[1]:


#Standard libaries for data analysis.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Using pandas setoption method to see all the columns in dataset.
pd.set_option('display.max_columns', None)


# In[3]:


df=pd.read_csv("D:/Python/Titanic Dataset/titanic_clean.csv")


# In[4]:


#Checking the dataframe
df


# In[5]:


# Dropping the unnamed column
df=df.drop(['PassengerId'],axis=1)


# In[6]:


df.isnull().sum()


# The data does not has any null values and all the unecessary columns have been dropped.

# In[7]:


#Dorpping columns that are not required.

df=df.drop(['Unnamed: 0'], axis=1)


# ### EDA

# In[8]:


# Using correlation to look for relationships b/w the features.
cor=df.corr()

cor


# In[9]:


sns.heatmap(cor,annot=True)


# Observation:
# 
# 1. The target variable has a positive correlation with Parch and Fare.
# 2. Fare is positively correlated with  every feature except Pclass.
# 3. Pclass is positvely correlated with Sibsp and Parch.
# 4. Age has positive correlation with Fare.

# In[10]:


#Making seperate list of continuous and discrete features.
dis_feature=['Survived','Pclass','SibSp','Parch','Embarked','Sex']
con_fearture=['Age','Fare']


# In[11]:


for i in dis_feature:
    sns.countplot(x=i,data=df,hue='Survived')
    plt.show()


# Based on the graph above, it is evident that
# 
# 1. The Titanic disaster claimed the majority of lives.
# 
# 2. Based on Pclass 3 being the lowest passenger class, the number of fatalities is highest, indicating that Pclass 1 and Pclass 2 passengers were saved first.
# 
# 3. It's also obvious that women were saved first.
# 
# 4. Sibsp (number of siblings/spouse aboard) and Parch (Parents and children) Based on these characteristics, we can observe that the majority of people who travelled alone died the most, and as the number of people travelling together increased, so did their survival rate.
# 
# 5. The majority of those who boarded the ship from Southampton died, according to the point of embarkation; this may be because the majority of Southhampton residents were Pclass-3 students.
# 

# In[12]:


for i in con_fearture:
    sns.histplot(x=df[i][df['Survived']==0],edgecolor="Black", color='pink',bins=20)
    sns.histplot(x=df[i][df['Survived']==1],edgecolor="Black", color='turquoise',bins=20)
    plt.show()


# This can be seen based on Age and Fare.
# 
# 1. The majority of fatalities were among those who paid the ticket price between 0-100.
# 2. In terms of age, it can be seen that children between the ages of 0 and 10 were rescued, while those who perished most commonly were between the ages of 17 and 40.

# In[13]:


sns.scatterplot(x='Parch',y='Pclass',data=df,hue="Survived")
plt.yticks([1,2,3])
plt.show()


# In[14]:


sns.scatterplot(x='Fare',y='Parch',hue='Survived',data=df)
plt.show()


# In[15]:


sns.scatterplot(x='Age',y='Parch',hue='Survived',data=df)
plt.show()


# When comparing Parch and Pclass, it is clear that people in the lowest class, 3, died the most, regardless of the number of parent and child travelling. People who died in Plclass 2 were mostly travelling alone, whereas people who died in Pclass 3 were mostly travelling with a group of four or more, implying that either the parent died saving their children or the children (adults) died saving their parents.
# 
# When comparing Parch with fare, it can be seen that the majority of the people who died purchased the ticket between $0-$100, which can also imply that the majority of them were in Pclass 3, regardless of the number of parent and children travelling.
# 
# Furthermore, the number of dead is higher if the parent is travelling without any child, assuming the age of the children is less than 25, so those who died are between the ages of 22 and 70.

# In[16]:


sns.scatterplot(x='SibSp',y='Pclass',data=df,hue="Survived")
plt.yticks([1,2,3])
plt.show()


# In[17]:


sns.scatterplot(x='Fare',y='SibSp',hue='Survived',data=df)
plt.show()


# In[18]:


sns.scatterplot(x='Age',y='SibSp',hue='Survived',data=df)
plt.show()


# When Pclass and Sibsp are compared, it is again clear that people travelling in Pclass 3 died the most, regardless of the number of parent and children or people travelling with their sibling.
# 
# Furthermore, in the case of SibSp, people who purchased tickets between 0 and 100 died the most, and the age of people who died is also the same as age and Parch, which is 22-70 years.

# In[19]:


df


# #### Preprocessing the Data

# ###### Splitting the dependent and independent variable

# In[20]:


x=df.drop(['Survived'],axis=1)


# In[21]:


x


# In[22]:


y=df["Survived"]


# In[23]:


y


# In[24]:


#### sklearn modules for data preprocessing:

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer


# In[25]:


### Using "OneHot Encoder" to sca;w down the categoliral variable and "StandardScaler" to scale the numerical data.
encoder=ColumnTransformer([('one',OneHotEncoder(),[1,6]),('sc',StandardScaler(),[0,2,3,4,5])],remainder='passthrough')


# In[26]:


# Passing dataframe 'x' to transform the data .
newdata=encoder.fit_transform(x)


# In[27]:


newdata


# As scalling the value we got the data  in an nd.array format we will make a dataframe of this array.

# In[29]:


z=pd.DataFrame(newdata)


# In[30]:


z


# In[89]:


#Now,splitting the data into train and test through sklearn library
#sklearn modules for Model Selection:

from sklearn.model_selection import train_test_split


xtrain,xtest,ytrain,ytest=train_test_split(z,y,train_size=0.70)


# In[90]:


print(xtrain.shape)
print(ytrain.shape)


# In[91]:


print(xtest.shape)
print(ytest.shape)


# #####  Developing the model to make prediction

# In[92]:


#Using logistic Regression to develop the model as the target variable is dicrete/categorical.

from sklearn.linear_model import LogisticRegression


# In[93]:


# Passing the x,y train data into the model
model=LogisticRegression() #Using an object to call the logistic regression class.
model.fit(xtrain,ytrain)


# ###### Making Predictions

# In[94]:


ypred=model.predict(xtest)

ypred


# ###### Checking for the accuracy of the model 

# In[95]:


##Importing library to check the accuracy of the model.
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics


# In[96]:


print(accuracy_score(ytest,ypred))
print(metrics.confusion_matrix(ytest,ypred))


# In[97]:


print(classification_report(ytest,ypred))


# ## Building a Random Forest model

# In[106]:


from sklearn.ensemble import RandomForestClassifier


# In[107]:


model1=RandomForestClassifier()
model1.fit(xtrain,ytrain)


# In[108]:


ypred1=model.predict(xtest)

ypred1


# In[109]:


#### Checking for the accuraccy of the model.
print(accuracy_score(ytest,ypred))
print(metrics.confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# ### Trying XgBoost to check if the f1-score and accuracy can be improved.

# In[110]:


#Now using xgboost for our prediction.
import xgboost
xgb=xgboost.XGBClassifier()


# In[111]:


xgb.fit(xtrain,ytrain)


# In[112]:


ypred2=xgb.predict(xtest)

ypred2


# In[113]:


#### Checking for the accuraccy of the model.
print(accuracy_score(ytest,ypred))
print(metrics.confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# From the above result and the accuracy report we have generated giving us the same value of the model accuracy.With the help of both the values we can say that the model is 82 percent accurate in classifying the dependent variable and all the three models are giving the same accuracy and f1-score.

# In[ ]:




