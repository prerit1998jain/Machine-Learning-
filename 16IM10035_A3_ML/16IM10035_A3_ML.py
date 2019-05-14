#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


data = pd.read_csv('AAAI.csv')


# In[3]:


topic = []
for i in range(len(data)):
    topic.append(data['Topics'][i].split("\n"))
#print(topic)


# In[4]:


def jaccard_coff(a,b):
    intersection = len(set(a).intersection(set(b)))
    union = len(set(a).union(set(a)))
    return(intersection/union)


# In[5]:


def maximum(matrix1):
    max1=-5
    k1=-1
    k2=-1
    for i in range(len(matrix1)):
        for j in range(len(matrix1)):
            if(matrix1[i][j]>=max1):
                max1=matrix1[i][j]
                k1=i
                k2=j
    return k1,k2 


# In[6]:


similarity_matrix = []
score = []
for i in (range(len(topic))):
    for j in range(len(topic)):
        if(i == j):
            score.append(-1)
        else:
            score.append(jaccard_coff(topic[i],topic[j]))
    similarity_matrix.append(score)
    score = []
    
print(np.shape(similarity_matrix))


# In[7]:


def complete_linkage(cluster1, cluster2):
    list = []
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            list.append(jaccard_coff(cluster1[i],cluster2[j]))
            #print(jaccard_coff(i,j))
    #print(len(list))
    return(min(list))
    


# In[8]:


def split1(list1):
    list2=[]
    list3=[]
    for i in range(len(list1)):
            list2.append(list1[i])
    list3.append(list2)
    return list3


# In[10]:


def single_linkage(cluster1, cluster2):
    list = []
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            list.append(jaccard_coff(cluster1[i],cluster2[j]))
            #print(jaccard_coff(i,j))
    #print(len(list))
    return(max(list))
        


# In[11]:


def hier_clust(topic,n, method):
    cluster = [k for k in range(len(topic))]
    while(len(topic)>n):
        cluster1 = []
        topic1 = []
        score = []
        mat = []
        #print(len(topic))
        for i in range(len(topic)):
            score = []
            for j in (range(len(topic))):
                if(i!=j):
                    if(method == 'single_linkage'):
                        value = (single_linkage(split1(topic[i]),split1(topic[j])))
                        score.append(value)
                    if(method == 'complete_linkage'):
                        value = (complete_linkage(split1(topic[i]),split1(topic[j])))
                        score.append(value)
                else:
                    score.append(-1)
            mat.append(score)
            max_i,max_j = maximum(mat)
            #print(score)
            score = []
        print(max_i)
        cluster1.append(max_i)
        topic1.append(topic[max_i])
        cluster1.append(max_j)
        topic1.append(topic[max_j])
        cluster.remove(max_i)
        topic.remove(topic[max_i])
        cluster.remove(max_j)
        topic.remove(topic[max_j])
        cluster.append(cluster1)
        topic.append(topic1)
        print(topic)
        print(len(topic))
    return(topic)
            
            
            


# In[13]:


cluster= hier_clust(topic,9,'single_linkage')


# In[ ]:




