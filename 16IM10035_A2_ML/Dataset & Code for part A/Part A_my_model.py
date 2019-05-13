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
train = list(np.concatenate((X_train,np.reshape((Y_train),(len(X_train),1))),axis = 1))
test = list(np.concatenate((X_test,np.reshape((Y_test),(len(X_test),1))),axis = 1))


##########################################################################
##########################################################################
## Writing Decision tree from scratch without using any python library  ##
##########################################################################
##########################################################################


### Creating a split on the basis of an attribute ###
def split_data(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index]==value:
            left.append(row)
        else:
            right.append(row)
    return( left, right)


### Computing
def gini_index(splits,values):
    total_count = sum([len(split) for split in splits])
    gini = 0.0
    for split in splits:
        size = len(split)
        if size == 0:
            continue
        score = 0
        for value in values:
            score += pow(([row[-1] for row in split].count(value)/size),2)
        gini += (1-score)*(size/total_count)
    return(gini)

def entropy(splits,values):
    total_count = sum([len(split) for split in splits])
    entropy = 0
    for split in splits:
        size = len(split)
        if size == 0:
            continue
        score = 0
        for value in values:
            entropy += ([row[-1] for row in split].count(value)/size)*np.log2(([row[-1] for row in split].count(value)/size))
        gain += -1*(entropy)*(size/total_count)
    return(gain)

def get_best_split_entropy(dataset): 
    y_label = list(set(row[-1] for row in dataset))
    best_index,best_value,best_score,best_splits = 99,99,99,None
    for index in range(len(dataset[0])-1):
        for i in range(2):
            splits = split_data(index,i,dataset)
            entropy = entropy(splits,y_label)
            if entropy > best_score:
                best_index = index
                best_value = i
                best_score = gini
                best_splits = splits
    return{'index':best_index,'value':best_value,'best_score':best_score,'splits':best_splits}


def get_best_split_gini(dataset): 
    y_label = list(set(row[-1] for row in dataset))
    best_index,best_value,best_score,best_splits = 99,99,99,None
    for index in range(len(dataset[0])-1):
        for i in range(2):
            splits = split_data(index,i,dataset)
            gini = gini_index(splits,y_label)
            if gini < best_score:
                best_index = index
                best_value = i
                best_score = gini
                best_splits = splits
    print("The best gini index for this node is", best_score)
    return{'index':best_index,'value':best_value,'best_score':best_score,'splits':best_splits}

    
def output_majority_class(split):
    y_label = [row[-1] for row in split]
    return(max(set(y_label), key = y_label.count))

def recursive_splitting_gini(node, max_depth,depth): 
    left, right = node['splits']
    del(node['splits'])
    if not left or not right:
        node['left'] = node['right'] = output_majority_class(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = output_majority_class(left),output_majority_class(right)
        return
    else:
        node['left'] = get_best_split_gini(left)
        recursive_splitting_gini(node['left'], max_depth,depth+1)
        node['right'] = get_best_split_gini(right)
        recursive_splitting_gini(node['right'], max_depth, depth+1)
        
        
def recursive_splitting_entropy(node, max_depth,depth): 
    left, right = node['splits']
    del(node['splits'])
    if not left or not right:
        node['left'] = node['right'] = output_majority_class(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = output_majority_class(left),output_majority_class(right)
        return
    else:
        node['left'] = get_best_split_entropy(left)
        recursive_splitting_entropy(node['left'], max_depth,depth+1)
        node['right'] = get_best_split_entropy(right)
        recursive_splitting_entropy(node['right'], max_depth, depth+1)

def build_tree_gini(train, max_depth):
    root = get_best_split_gini(train)
    recursive_splitting_gini(root, max_depth,1)
    return(root)

def build_tree_entropy(train, max_depth):
    root = get_best_split_entropy(train)
    recursive_splitting_entropy(root, max_depth,1)
    return(root)
    
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s|X%d = %d' % ((depth*'\t', (node['index']+1), int(node['value']))))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s:[%s]' % ((depth*'\t', node)))
    

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def prediction(test,tree):
    y_pred = []
    for row in test:
        y_pred.append(predict(tree,row))
    return(y_pred)



############
tree_gini = build_tree_gini(train,4)
print_tree(tree_gini)
type(test)
y_pred = prediction(test,tree_gini)
cal_accuracy(y_pred,Y_test)

