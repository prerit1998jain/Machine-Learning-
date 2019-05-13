import numpy as np
from sklearn.metrics import confusion_matrix
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# Reading the txt file and storing the data into a list of list.
def read_dataset(file_X,file_Y):
    list_of_lists = []
    with open(file_X,'r') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    data = np.asarray(list_of_lists,dtype = int)
    X_train = np.zeros(np.max(data,axis = 0))
    for i in range(len(data)):
        for j in range(1):
            doc_id = data[i][j]
            word_id = data[i][j+1]
            X_train[doc_id-1][word_id-1]=1
    train_label = open(file_Y,'r').readlines()
    Y_train = np.asarray(train_label,dtype = int)
    return(X_train,Y_train)

X_train,Y_train = read_dataset('traindata.txt','trainlabel.txt')
X_test, Y_test = read_dataset('testdata.txt','testlabel.txt')
# Equating the number of columns in test and train set
X_test = X_test[:,0:np.shape(X_train)[1]]

train = list(np.concatenate((X_train,np.reshape((Y_train),(len(X_train),1))),axis = 1))
test = list(np.concatenate((X_test,np.reshape((Y_test),(len(X_test),1))),axis = 1))


# Using Scikit Learn Library for classifcation
def train_using_gini(X_train, y_train, max_depth):
    # Creating the classifier object 
    dectree_gini = DecisionTreeClassifier(criterion = "gini",max_depth= max_depth) 
    # Performing training 
    dectree_gini.fit(X_train, y_train) 
    return dectree_gini

# Using Entropy
def train_using_entropy(X_train,y_train,max_depth):
    dectree_entropy = DecisionTreeClassifier(criterion = 'entropy',max_depth = max_depth)
    dectree_entropy.fit(X_train,y_train)
    return(dectree_entropy)

# Prediction on the basis of the model
def prediction(X_test,model):
    y_pred = model.predict(X_test)
    return(y_pred)

#calculating accuracy and other metrics
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n ", confusion_matrix(y_test,y_pred))
    print("Accuracy: \n", accuracy_score(y_test,y_pred))
    print("Classification_Report: \n", classification_report(y_test,y_pred))


# Function for Decision tree using scikit learn package from python
def DecisionTree_using_scikitlearn(X_train,Y_train,method,max_depth):
    if method == 'gini_index':
        dec_tree_gini = train_using_gini(X_train, Y_train,max_depth)
        return(dec_tree_gini)
    if method == 'entropy':
        dec_tree_entropy = train_using_entropy(X_train, Y_train,max_depth)
        return(dec_tree_entropy)


print("################################################################# \n",
     "The following results are obtained using the scikit learn package \n",
     "################################################################## \n")


print("The predicted values through decision tree using gini index as a crtieria \n")
y_pred_gini = prediction(X_test, DecisionTree_using_scikitlearn(X_train,Y_train,'gini_index',5))
cal_accuracy(Y_test,y_pred_gini)
    
print("The predicted values through decision tree using entropy as a crtieria \n")
y_pred_entropy = prediction(X_test, DecisionTree_using_scikitlearn(X_train,Y_train,'entropy',5))
cal_accuracy(Y_test,y_pred_entropy)


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
        entropy_score = 0
        gain = 0
        for value in values:
            entropy_score += ([row[-1] for row in split].count(value)/size)*np.log2(([row[-1] for row in split].count(value)/size))
        gain += -1*(entropy_score)*(size/total_count)
    return(gain)

def get_best_split_entropy(dataset): 
    y_label = list(set(row[-1] for row in dataset))
    best_index,best_value,best_score,best_splits = 99,99,99,None
    for index in range(len(dataset[0])-1):
        for i in range(2):
            splits = split_data(index,i,dataset)
            entropy_score = entropy(splits,y_label)
            if entropy_score > best_score:
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
    print("Gini Index of best split is", best_score)
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
tree_gini = build_tree_gini(train,3)
print_tree(tree_gini)
type(test)
y_pred = prediction(test,tree_gini)
cal_accuracy(y_pred,Y_test)


