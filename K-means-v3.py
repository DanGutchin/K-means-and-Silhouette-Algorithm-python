import random
import numpy as np
from time import time
from sklearn.cluster import KMeans

def distMat(dataMatrix, centers):   
    data_rows = dataMatrix.shape[0]
    centers_rows = centers.shape[0]
    sum_of_xi = np.sum((dataMatrix**2),axis=1).reshape([data_rows,1])
    sum_of_yi = np.sum((centers**2),axis=1).reshape([1,centers_rows]) 
    prod_of_data_centers = dataMatrix @ centers.T
    return sum_of_xi - 2*prod_of_data_centers + sum_of_yi

def findCenters(dataMatrix, centers):
    return distMat(dataMatrix, centers).argmin(axis=1)

def probability_selection(distmat):
    probabilities = distmat/distmat.sum()
    probabilities[probabilities<0] = 0
    return probabilities

def initCenters(dataMatrix, k):
    centers = [dataMatrix[random.randint(0,dataMatrix.shape[0] - 1),:]]
    for i in range(k-1):
        distmat = distMat(dataMatrix, np.array(centers))
        mindist = distmat.min(axis=1)
        probabilities = probability_selection(mindist)
        index = np.random.choice(dataMatrix.shape[0],1,p=probabilities)[0]
        centers.append(dataMatrix[index,:])
    return np.array(centers)

def kmeansPP(dataMatrix, k):
    data_rows = dataMatrix.shape[0]
    centers = initCenters(dataMatrix, k)
    old_clusters = np.full([data_rows],0)
    while True:
        new_clusters = findCenters(dataMatrix, centers)
        if (old_clusters == new_clusters).all():
            return new_clusters
        old_clusters = new_clusters
        centers = [np.mean(dataMatrix[new_clusters == i,:], axis=0)for i in range(k)]
        centers = np.array(centers)
        
def silhouette(dataMatrix, clusters, k):
    data_rows = dataMatrix.shape[0]
    centers = [np.mean(dataMatrix[clusters == i,:],axis=0)for i in range(k)]
    centers = np.array(centers)

    dist = distMat(dataMatrix,centers)    
    a = np.empty([data_rows])
    b = np.empty([data_rows])
    
    for i in range(data_rows):
        a[i] = dist[i,clusters[i]]
        b[i] = (dist[i,:])[np.arange(k)!=clusters[i]].min()
    
    return ((b-a)/np.maximum(a,b)).mean()

############################### run ##########################################

dataMatrix= np.loadtxt("./data_1_3.txt", delimiter = ",")
max_kmeans = 13

tt = time()
y = kmeansPP(dataMatrix,10)
silhouette(dataMatrix,y,10)
print(time() - tt)

values = [kmeansPP(dataMatrix,i)for i in range(2,max_kmeans)]
silo = [silhouette(dataMatrix,values[i],i+2)for i in range(max_kmeans-2) ]
best_score = np.argmax(np.array(silo))
print("the best K for this data is: ",best_score+2)
print("time for running K times Kmeans++: ",time() - tt)

tt = time()
kmeans = KMeans(n_clusters=10, random_state=0).fit(dataMatrix)
print(time() - tt)