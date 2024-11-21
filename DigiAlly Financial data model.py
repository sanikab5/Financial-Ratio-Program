#!/usr/bin/env python
# coding: utf-8

# # DigiAlly Sample program

# ### Importing Basic Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df= pd.read_csv('Approach 2.0.csv')
df


# In[4]:


df.head(5)


# ### Basic EDA (Exploratoty Data Analysis)

# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# ### Performing Test Train model on Unstandardized data

# In[8]:


import sklearn


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Final Band'], df['Final Band'], stratify=df['Final Band'], random_state=66)
from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# ### Accuracy of Non- Standardized model 

# In[10]:


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))


# In[11]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=46)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# In[12]:


tree = DecisionTreeClassifier(max_depth=3, random_state=66)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# In[13]:


print("Feature importances:\n{}".format(tree.feature_importances_))


# ### The Uneven Distributed Data Visualization

# In[14]:


import seaborn as sns
sns.countplot(x=df['Final Band'])


# ### Installing Synthetic Data Vault lib

# In[16]:


import sdv


# ### Generating Synthetic Data

# In[3]:


from sdv.tabular import GaussianCopula

# Load your real dataset
data = pd.read_csv('Approach 2.0.csv')

# Initialize and fit the GaussianCopula model
model = GaussianCopula()
model.fit(data)

# Generate synthetic data
synthetic_data = model.sample(len(data))
synthetic_data
# Export the synthetic data to a CSV file
#synthetic_data.to_csv('D:/Downloads/ap2.csv', index=False)


# ### Comparing sd and df

# In[18]:


sd= pd.read_csv('D:/Downloads/ap2.csv')
sd.head()


# In[19]:


df.head()


# ### Using SMOTE to balance the data

# In[20]:


from imblearn.over_sampling import SMOTE
import pandas as pd

# Loading the imbalanced dataset
data = pd.read_csv('D:/Downloads/ap2.csv')

# Separate the features and target variable
X = data.drop(['Final Band'], axis=1)
y = data['Final Band']

# Initialize and fit the SMOTE object
smote = SMOTE()
d1= X_resampled, y_resampled = smote.fit_resample(X, y)

# Print the class distribution before and after applying SMOTE
print('Class distribution before SMOTE:')
print(y)
print(y.value_counts())

print('Class distribution after SMOTE:')
print(y_resampled.value_counts())
print(y_resampled)


# ### Accuracy

# In[21]:


import seaborn as sns
sns.countplot(x=y_resampled)


# In[22]:


df1=pd.DataFrame(X_resampled)
df1['y_resampled']= y_resampled
df1


# ### Accuray with linear regression

# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df1[['Revenue Growth','PAT margin','ROE','Debt to Equity','Interest Coverage','Cash Conversion','Current Ratio','Return on capital Employed']], df1['y_resampled'], test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Use the model to make predictions on the testing data
predictions = model.predict(X_test)

# Evaluate the model's performance
score = model.score(X_test, y_test)
print("Model score: ", score)


# ### Accuracy using Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate a random dataset for demonstration
X_train, X_test, y_train, y_test = train_test_split(df1[['Revenue Growth','PAT margin','ROE','Debt to Equity','Interest Coverage','Cash Conversion','Current Ratio','Return on capital Employed']], df1['y_resampled'], test_size=0.25, random_state=42)

# Create a Random Forest Classifier with trees
rfc = RandomForestClassifier(n_estimators=1000, random_state=49)

# Train the model on the training data
rfc.fit(X_train, y_train)

# Use the trained model to predict on the test data
y_pred = rfc.predict(X_test)

# Evaluate the model's performance
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:




