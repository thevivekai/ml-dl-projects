# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:43:16 2025

@author: Vivek Prakash Upreti
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#customer_data = pd.read_csv(r'C:\Users\Vivek Prakash Upreti\Desktop\AI_ML\Unsupervised Learning\Hierarchical Clustering\Mall_Customers.csv')
customer_data = pd.read_csv('C:/Users/Vivek Prakash Upreti/Desktop/AI_ML/Unsupervised Learning\Hierarchical Clustering/Mall_Customers.csv')
customer_data

# Extract the useful column from original data i.e last 2 column

data=customer_data.iloc[:,3:5]

# scipy library is useful to plot  dendogram directly
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10,7))
plt.title("mall Customer Dendogram")
dend = sch.dendrogram(sch.linkage(data,method='ward'))

# dendogram with scikit library
from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=5,linkage='ward')

labels=cluster.fit_predict(data)
labels

plt.figure(figsize=(10,7))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'],c=cluster.labels_,cmap='rainbow')