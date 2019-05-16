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
    return(np.matmul(np.transpose(x),(hypo_func(x,beta)-y))*(1/np.size(y)))


# Defining Gradient Descent Algorithm
def grad_desc(beta,x,y,learn_rate,max_iter):
    loss = np.zeros(max_iter)
    for i in range(max_iter):
        temp = beta - learn_rate*dMSE(y,x,beta)
        beta = temp
        y_est = hypo_func(x,beta)
        #print(dMSE(y,x,beta))
        loss[i] = MSE_loss(y,y_est)
        #print("Loss at iteration",i,"is",loss[i])
    return(beta,loss)
        

# Linear Regression Model
def Lin_reg_train(x_train,y_train,learn_rate,max_iter,order):
    beta = wt_init(order)
    beta_final,loss = grad_desc(beta,x_train,y_train,learn_rate,max_iter)
    print("Final training set loss value is",np.min(loss))
    return(beta_final,np.min(loss))

# Prediction function
def lin_reg_predict(x_test,y_test,beta_final):
    y_est = hypo_func(x_test,beta_final)
    loss = MSE_loss(y_test,y_est)
    print("The loss function value on test set is", MSE_loss(y_test,y_est))
    return(y_est,loss)
    
# Ans - 2 & 3(a)
# Plotting the dataset 
# Synthetic Dataset Plotting
# Use plot_curves function to plot the the fitted curve for desired dataset and desired order
def plot_curves(size,order):
    dataset = data_gen(size,order)
    x_train,y_train,x_test,y_test = split_dataset(dataset,0.2)
    beta_final = np.ones(order+1)
    beta_final,loss = Lin_reg_train(x_train,y_train,0.05,2000,order)
    #scipy.optimize.curve_fit(f,dataset[1],dataset[2])
    dataset_plot = dataset.sort_values(1,ascending = True)
    plt.plot(dataset_plot[1],dataset_plot['y'])
    def data_for_plot(size,order):
        X = np.linspace(0,.9,size)
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
    data = data_for_plot(20,order).sort_values(1,ascending = True)
    x = (data.loc[:,data.columns != 'y'])
    x = x.values
    y = hypo_func(x,beta_final)
    plt.plot(x[:,1],y)
# Give desired arguments for the plots of respective order and sizes. I have used this code in order to generate the plots
plot_curves(10000,9)
    

# Using this code I have plotted the training and test error with degree of polynomial fitted
# Ans - 2 & 3 (b)

loss_train = np.ones(9)
loss_test = np.ones(9)
size = 10000
for order in range(9):
    dataset = data_gen(size,order)
    x_train,y_train,x_test,y_test = split_dataset(dataset,0.2)
    beta_final = np.ones(order+1)
    beta_final,loss = Lin_reg_train(x_train,y_train,0.05,2000,order)
    loss_train[order] = loss
    y_est,loss_t = lin_reg_predict(x_test,y_test,beta_final)
    loss_test[order] = loss_t
print(loss_train)
plt.plot(loss_train, label = 'train_set')
plt.title(("Dataset size :", size),)
plt.plot(loss_test,label= 'test_set')
plt.legend()

plt.xlabel('Degree')
plt.ylabel('loss')
plt.plot






