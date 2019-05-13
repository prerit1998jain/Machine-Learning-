import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('train.csv', sep= ',', header = None)
test = pd.read_csv('test.csv', sep= ',', header = None)
train = (train[1:len(train)][:])
test = (test[1:len(test)][:])
train = train.apply(LabelEncoder().fit_transform)
test = test.apply(LabelEncoder().fit_transform)
#print(train)


X_train = train.values[:,0:3]
#print(X_train)
Y_train = train.values[:,4]
X_test = test.values[:,0:3]
Y_test = test.values[:,4]
train = list(train)
test = list(test)

# Using Scikit Learn Library for classifcation
def train_using_gini(X_train, y_train):
    # Creating the classifier object 
    dectree_gini = DecisionTreeClassifier(criterion = "gini") 
    # Performing training 
    dectree_gini.fit(X_train, y_train) 
    return dectree_gini

def train_using_entropy(X_train,Y_train):
    dectree_entropy = DecisionTreeClassifier(criterion = "entropy")
    dectree_entropy.fit(X_train,Y_train)
    return(dectree_entropy)

def prediction(X_test, model):
    y_pred = model.predict(X_test)
    print("Predicted Values:")
    print(y_pred)
    return(y_pred)


def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n ", confusion_matrix(y_test,y_pred))
    print("Accuracy: \n", accuracy_score(y_test,y_pred))
    print("Classification_Report: \n", classification_report(y_test,y_pred))


dec_tree_gini = train_using_gini(X_train, Y_train)
print("The predicted values through decision tree using gini index as a crtieria \n")
y_pred_gini = prediction(X_test, dec_tree_gini)
cal_accuracy(Y_test,y_pred_gini)


dec_tree_entropy = train_using_entropy(X_train, Y_train)

print("The predicted values through decision tree using entropy as a crtieria \n")
y_pred_entropy = prediction(X_test, dec_tree_entropy)
#print(train.apply(LabelEncoder().inverse_transform))

cal_accuracy(Y_test,y_pred_entropy)




