
# %%
#Importing packages
from turtle import color
import pandas as pd
import numpy as np
import seaborn as sns
import pydot 
import matplotlib.pyplot as plt

from io import StringIO
from sklearn import metrics
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# %%
#Load the dataset
data = pd.read_csv("diabetes.csv")

#Exploratiry Data Analysis
#Structure Investigation
print("Head of data:\n", data.head()) #Head of data
print(data.info()) #Data Information
print("Data Description:\n", data.describe()) #Data description
print("Data type count:\n", pd.value_counts(data.dtypes)) #Data type count
uniqueItems = data.select_dtypes(include="number").nunique().sort_values() #Unique datatypes
uniqueItems.plot.bar(logy=True, figsize=(15,5), title="Unique Items per feature") #plotting unique values

#%%
#Quality Investigation
duplicates = data.duplicated().sum()
print("Found", str(duplicates), "Duplicates")

## NA values
print("NA Values:", data.isnull().values.any())

## Zero Values
columns = data.columns[:-1]
print("Total number of rows:", len(data))
for i in columns:
    print("Number of zero's in column", i, ":", (data[i]==0).sum())

# Replacing Zero with Median values
print("Replacing Zero values of Columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI")
for i in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    data[i] = data[i].map(lambda x: round(data[i].median(),2) if x==0 else x)

data['Outcome'].hist() #Plotting histogram for output value

#%%
#Content Investigation
#Pairplot for detailed feature evaluation
sns.pairplot(data, hue='Outcome', diag_kind='kde')
#Heatmap to plot correlation
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data', fontsize=13)
plt.show()

# %%
# Data Preprocessing
X = data.drop("Outcome", axis=1).values #Selecting features
y = data["Outcome"].values #Selecting Target

# Feature Scaling
sc = StandardScaler() #Initilaising standard scaler 
X = sc.fit_transform(X) #Transforming data to standard format (0 to 1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# %%
# K-NN
#Elbow-Method Exploring for choosing K Value
error = []
for k in range(1,30):
    #Initialise KNN with euclidean distance
    knn = KNeighborsClassifier(n_neighbors=k, p=2) 
    knn.fit(X_train, y_train) #Fit train data
    # Predicting the Test set results
    y_pred = knn.predict(X_test) #Predicting using fitted model
    error.append(np.mean(y_pred != y_test)) #Finding errors

# K-Fold Cross validation
scores = []
for k in range(1,30):
    #Initialise KNN with euclidean distance
    knn = KNeighborsClassifier(n_neighbors = k, p=2) 
    cv = KFold(n_splits=4, random_state=1, shuffle=True) #K-Fold cross validation
    score = cross_val_score(knn, X=X_train, y=y_train, cv=cv, scoring="accuracy")
    scores.append(1-score.mean()) # Finding average accuracy score for each K

def plotKNN(metric, title):
    plt.figure(figsize=(10,6))
    plt.plot(range(1,30), metric, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
    plt.title(title)
    plt.xlabel('K')
    plt.ylabel('Error Rate')

#Plot Error_rate using Elbow Method
plotKNN(error, 'Elbow Method - Error Rate vs. K Value')
#Plot Error_rate using K-fold corss validation method
plotKNN(scores, 'K-Fold CV Method - Error Rate vs. K Value')

# Using Elbow method, choosing Final value of K
k = 9
#Initialise KNN with euclidean distance and K value
knn = KNeighborsClassifier(n_neighbors=k, p=2)
knn.fit(X_train,y_train) #Fit train data
y_pred = knn.predict(X_test) #Predict using test data
print('For K=', k, '\n')
#Train Accuracy
print("Train Accuracy is", round(accuracy_score(y_train, knn.predict(X_train))*100, 2), "%")
#Confusion Matrix
print("Confusion matrix:\n ", confusion_matrix(y_test, y_pred))
#Classification Report
print("Classification Report:\n ", classification_report(y_test,y_pred))
# %%
# Decision Tree
print("Base Model for Decision Tree")
dtree = DecisionTreeClassifier() #Initialise Decision tree 
dtree.fit(X_train,y_train) #Fit train data
y_pred = dtree.predict(X_test) #Predict using test data
print("Confusion matrix:\n ", confusion_matrix(y_test, y_pred)) #Confusion Matrix
print("Classification Report:\n ", classification_report(y_test,y_pred))

#Plot Decision tree
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data, feature_names=data.columns[:-1],filled=True,rounded=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())

#%%
# Using GirdSearchCV for Hyperparameter tuning
if False: #Set True for grid search
    # combinations of parameters for grid search
    params = { 'criterion': ['gini', 'entropy'], 'max_depth' : range(2,20,1),
                'min_samples_leaf' : range(1,20), 'min_samples_split': range(2,20),
                'splitter' : ['best', 'random'] }
    #Grid search with crossvalidation of 4 folds
    grid_search = GridSearchCV(estimator=dtree, param_grid=params, cv=4, n_jobs =-1)
    grid_search.fit(X_train,y_train) #Fit train data
    best_param = grid_search.best_params_ #Use best parameters
    print(best_param)

# Using best parameters predicted by GridSearchCV
print("Fine tuned Final model for Decision tree")
dtree = DecisionTreeClassifier(criterion='gini', splitter='best', 
         max_depth=16, min_samples_split=12, min_samples_leaf=15)
dtree.fit(X_train,y_train) #Fit train data
y_pred = dtree.predict(X_test) #Predict using test data
#Train Accuracy
print("Train Accuracy is", round(accuracy_score(y_train, dtree.predict(X_train))*100, 2), "%")
print("Confusion matrix:\n ", confusion_matrix(y_test, y_pred)) #Confusion Matrix
print("Classification Report:\n ", classification_report(y_test,y_pred)) #Classification report

#Plot Decision tree
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data, feature_names=data.columns[:-1],filled=True,rounded=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())

# %%
# Logistic Regression
print("Base Model for Logistic Regression")
logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)
y_pred = logistic_model.predict(X_test)

#Train Accuracy
print("Train Accuracy is", round(accuracy_score(y_train, logistic_model.predict(X_train))*100, 2), "%")
print("Confusion matrix:\n ", confusion_matrix(y_test, y_pred)) #Confusion Matrix
print("Classification Report:\n ", classification_report(y_test,y_pred)) #Classification report

# %%
if False:  #Set True for grid search
    grid={"C": np.logspace(-5, 8, 15), "penalty":['l1', 'l2', 'elasticnet'], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}
    logreg_cv=GridSearchCV(logistic_model, grid, scoring='accuracy', n_jobs=-1,cv=4)
    logreg_cv.fit(X_train,y_train)
    print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print("accuracy :", logreg_cv.best_score_)

# Using best parameters predicted by GridSearchCV
print("Fine tuned Final model for Logistic Regression")
logistic_model2 = LogisticRegression(C=0.05179474679231213, solver='newton-cg')
logistic_model2.fit(X_train,y_train)
y_pred = logistic_model2.predict(X_test)

#Train Accuracy
print("Train Accuracy is", round(accuracy_score(y_train, logistic_model2.predict(X_train))*100, 2), "%")
print("Confusion matrix:\n ", confusion_matrix(y_test, y_pred)) #Confusion Matrix
print("Classification Report:\n ", classification_report(y_test,y_pred)) #Classification report
# %%
#define metrics
y_pred_proba = logistic_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr, tpr, label="AUC="+ str(round(auc,2)))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
# %%