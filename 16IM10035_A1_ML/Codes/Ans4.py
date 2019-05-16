## Answer 4(a)


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy

# Dataset Creation
def data_gen(size,order):
    X = np.random.uniform(0,1,size)
    x = np.zeros((order+1,size))
    for i in range(order+1):
        for j in range(size):
            x[i][j] = np.power(X[j],i)
    Y = np.zeros(size)
    Y = np.sin(2*np.pi*X) + np.random.normal(0,0.3)
    Y = np.reshape(Y,(size,1))
    df = np.concatenate((x,np.transpose(Y)),axis = 0)
    df = pd.DataFrame(np.transpose(df))
    df = df.rename(columns = {order+1:'y'})
    return(df)


# Splitting the Dataset
def split_dataset(df,ratio):
    train, test = train_test_split(df, test_size = ratio)
    x_train = (train.loc[:,train.columns != 'y'])
    x_train = x_train.values
    y_train = (train['y']).values
    y_train = np.reshape(y_train,(np.size(y_train),1))
    x_test = test.loc[:, test.columns != 'y']
    x_test = x_test.values
    y_test =(test["y"]).values
    y_test = np.reshape(y_test,(np.size(y_test),1))
    return(x_train,y_train,x_test,y_test)

def wt_init(order):
    beta = np.random.uniform(0,1,(order+1,1))
    return(beta)

#Defining Hypothesis Function
def hypo_func(x,beta):
    return(np.matmul((x),beta))


# Defining MSE Loss function
def MSE_loss(y, y_est):
    MSE = np.mean((np.power((y-y_est),2)))
    return(MSE)
    

#Derivative of Loss Function
def dMSE(y,x,beta):
    return(np.matmul(np.transpose(x),np.power((hypo_func(x,beta)-y),1))*(1/np.size(y)))

def Power_4_loss(y,y_est):
    return(np.mean((np.power((y-y_est),4))))

def dPower_4_loss(y,x,beta):
    return(np.matmul(np.transpose(x),np.power((hypo_func(x,beta)-y,3)))*(1/np.size(y)))

# Defining Gradient Descent Algorithm
def grad_desc_MSE(beta,x,y,learn_rate,max_iter):
    loss = np.zeros(max_iter)
    for i in range(max_iter):
        temp = beta - learn_rate*dMSE(y,x,beta)
        beta = temp
        y_est = hypo_func(x,beta)
        #print(dMSE(y,x,beta))
        loss[i] = MSE_loss(y,y_est)
        #print("Loss at iteration",i,"is",loss[i])
    return(beta,loss)
        

def grad_desc_Power4(beta,x,y,learn_rate,max_iter):
    loss = np.zeros(max_iter)
    for i in range(max_iter):
        temp = beta - learn_rate*dPower_4_loss(y,x,beta)
        beta = temp
        y_est = hypo_func(x,beta)
        loss[i] = Power_4_loss(y,y_est)
        #print("Loss at iteration",i,"is",loss[i])
    return(beta,loss)
        

# Linear Regression Model
def Lin_reg_train(x_train,y_train,learn_rate,max_iter,order):
    beta = wt_init(order)
    beta_final,loss = grad_desc_Power4(beta,x_train,y_train,learn_rate,max_iter)
    print("Final training set loss value is",np.min(loss))
    return(beta_final,np.min(loss))

# Prediction function
def lin_reg_predict(x_test,y_test,beta_final):
    y_est = hypo_func(x_test,beta_final)
    loss = Power_4_loss(y_test,y_est)
    print("The loss function value on test set is", MSE_loss(y_test,y_est))
    return(y_est,loss)
    
# Execution 
def Execute(order):
    dataset = data_gen(10,order)
    final_loss = np.ones(9)
    x_train,y_train,x_test,y_test = split_dataset(dataset,0.2)
    print("Polynomial Model with degree", order)
    model_beta,loss_train = (Lin_reg_train(x_train,y_train,0.05,2000,order))
    print("These are the learned coffiecient matrix \n")
    print(model_beta)
    y_est,loss = lin_reg_predict(x_test,y_test,model_beta)
    return(loss_train)


#type(model_beta)
loss_train = np.zeros(9)
loss_test = np.zeros(9)
for i in range(9):
    loss_train = Execute(i+1)


