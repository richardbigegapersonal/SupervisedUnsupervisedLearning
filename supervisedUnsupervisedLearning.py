from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

iris=datasets.load_iris()
print(type(iris))
print(iris.keys())

print(type(iris.data), type(iris.target))
print("Shape of the Input: ",iris.data.shape, " Shape of the target: ", iris.target.shape )
print(iris.target_names)

x=iris.data;y=iris.target
df=pd.DataFrame(x,columns=iris.feature_names)
print(df.head())

_=pd.plotting.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='D')

#KNN - K Nearest Neibhors
#
#The basic idea of to predict a data label by
#
#Looking at the "K" closet label data points and getting them to vote on what label the unlabelled data should have.
#Scikit-Learn Fit and Predict
#
#All scikit-Learn machine models are implemented as python classes
#
#They implement the argorithms for learning and predicting
#They store all the information that is learned from the data
#Training a data on the model is also called fiting a data to the model.
#
#In scikit-learn we use the .fit() method to do this
#We use .predict method to predict a new and unlebeled data point
#Using Scikit-Learn to Fit a Classifier

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'],iris['target'])
x_new=np.array([[3.1,5,4,4.2],[8.1,5,9.2,6.0],[3.1,4,4.2,6.0],[6,6,4.2,6.0]])
prediction=knn.predict(x_new)
print(prediction)
print(type(iris.feature_names))
# Python3 code to iterate over a list
# Using for loop
for i in prediction:
    print(iris.target_names[i])
    
#**Measuring Model Performance**
    
from sklearn.model_selection import train_test_split
X=iris['data']
y=iris['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21, stratify=y)
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(y_pred)
print(knn.score(X_test,y_test))

# Import necessary modules
digits=datasets.load_digits()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Create feature and target arrays
X = np.array(digits.data)
y = np.array(digits.target)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))