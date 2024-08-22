from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.stats import mode
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation

"""
Apply three clustering techniques to the handwritten digits dataset. Assume that k = 10. (30
points)
a. K-means clustering (implemented in Problem 1).
b. Agglomerative clustering with Ward linkage (sklearn.cluster.AgglomerativeClustering).
c. Affinity Propagation (sklearn.cluster.AffinityPropagation).
The dataset you will be working with is the handwritten digits and the details can be found here.
Assess all three clustering algorithms using the following protocol:

i. Each cluster should be defined by the digit that represents the majority of the current cluster.
For examples, if in the second cluster, there are 60 data points of digit “5”, 40 of “3” and 25 of
“2”, the cluster is labeled as “5”.

ii. Report the 10x10 confusion matrix by comparing the predicted clusters with the actual labels
of the datasets. If the clustering procedure resulted in less than 10 clusters, output “-1” in the
position to the missing clusters in the confusion matrix.

iii. Calculate the accuracy of each clustering method using Fowlkes-Mallows index
"""
#load the data
digits = load_digits()
data = digits.data
target = digits.target

k = 10 #number of clusters

#function to calculate euclidean distance
def find_euclid_distance(i,j):
    return np.sqrt(np.sum((j-i) ** 2)) #equation for Euclidean distance


#K means function (from problem 1)
#takes in the data as an array, the number of clusters, k, and max number of times to run the function (if the function doesn't terminate on its own)
def k_means(data, k, max_iterations):
    centers = data[np.random.choice(range(len(data)), k, replace=False)] #randomly select k initial centers for the data

    for _ in range(max_iterations): #running the following code until max iterations or until there's no change in the clusters
        distances = []
        for element in data:
            distances.append([find_euclid_distance(element, center) for center in centers]) #calculating the distance of each element to the center of each cluster

        labels = np.argmin(distances, axis=1) #updating the labels of the element based on the shortest distance

        new_center = []
        for i in range(k): #calculating the new cluster center by finding the mean of that cluster
            cluster_mean = np.mean(data[labels == i], axis=0)
            new_center.append(cluster_mean)
        new_center = np.array(new_center) #updating the new cluster centers based on the mean values

        if np.all(centers == new_center): #checking if there were no changes in the cluster centers --> if so, end the function
            break

        centers = new_center #if there was a change in cluster centers, update the cluster centers and use those for the next iteration

    return labels


cluster_digits_k_means = k_means(data, k, 1000) #running my k means function
k_means_labels = cluster_digits_k_means.copy() #get a copy to keep the original labels

#ensure that the clusters are defined by the majority digit in each cluster
majority_digits = np.zeros(k, dtype=int) #set the majority digit to be 0
# loop through the clusters to find the majority digit of each
for cluster in range(k):
    cluster_index = np.where(k_means_labels == cluster)[0] #find the index of the cluster
    cluster_digits = target[cluster_index] #finding the elements of the cluster
    majority_digit = mode(cluster_digits).mode.item() #finding the majority (mode) digit
    majority_digits[cluster] = majority_digit #set the label to be the majority digit

majority_labels = majority_digits[k_means_labels] #create array with the majority labels

k_means_confusion_matrix = confusion_matrix(target, majority_labels) #creating a confusion matrix comparing the actual clusters with the predicted clusters

# If the clustering resulted in less than 10 clusters, fill the missing positions with -1
num_clusters = len(np.unique(k_means_labels)) #finding how many unique clusters there are (ie the number of clusters)
if num_clusters < k:
    expected_clusters = list(range(k)) #creating a list with the correct number of clusters
    unique_clusters = np.unique(k_means_labels) #getting the unqiue labels (labels for the clusters we do have)

#if the cluster does not have a unique label it was not created during clusters --> we need to assign it a -1
    for cluster in expected_clusters:
        if cluster not in unique_clusters:
            k_means_confusion_matrix = np.insert(k_means_confusion_matrix, cluster, -1, axis=1) #assign the missing cluster to be -1

print("K means confusion matrix using my algorithm: \n", k_means_confusion_matrix)
"""
K means confusion matrix using my algorithm: 
 [177   0   0   0   1   0   0   0   0   0]
 [  0  55  24   1   0   1   2   0  99   0]
 [  1   2 148  13   0   0   0   3   8   2]
 [  0   0   1 158   0   2   0   6   7   9]
 [  0   7   0   0 163   0   0   9   2   0]
 [  0   0   0   1   2 130   1   0   0  48]
 [  1   1   0   0   0   0 177   0   2   0]
 [  0   2   0   0   0   5   0 170   2   0]
 [  0   5   3   2   0   9   2   2 100  51]
 [  0  20   0   6   0   7   0   8   0 139]


 """
k_means_accuracy = metrics.fowlkes_mallows_score(target, majority_labels) #using the sklearn method to calculate the score
print("k means accuracy: ", k_means_accuracy)
#k means accuracy using my algorithm: 0.6966496162869433


#k means clustering using sklearn- I wasn't sure if I should use the sci-kit learn algorithm or my code from problem 1 so i included both
kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(data) #perform k means clustering

labels_k_sklearn = kmeans.labels_.copy() #get a copy of the cluster labels to keep the original labels

#ensure that the clusters are defined by the majority digit in each cluster
majority_digits = np.zeros(k, dtype=int) #set the majority digit to be 0
# loop through the clusters to find the majority digit of each
for cluster in range(k):
    cluster_index = np.where(labels_k_sklearn == cluster)[0] #find the index of the cluster
    cluster_digits = target[cluster_index] #finding the elements of the cluster
    majority_digit = mode(cluster_digits).mode.item() #finding the majority (mode) digit
    majority_digits[cluster] = majority_digit #set the label to be the majority digit

majority_labels = majority_digits[labels_k_sklearn] #create array with the majority labels
k_means_sklearn_confusion_matrix = confusion_matrix(target, majority_labels) #creating a confusion matrix comparing the actual clusters with the predicted clusters

# If the clustering resulted in less than 10 clusters, fill the missing positions with -1
num_clusters = len(np.unique(labels_k_sklearn))#finding how many unique clusters there are (ie the number of clusters)
#with k=10 there should be exactly 10 clusters
if num_clusters < k:
    expected_clusters = list(range(k)) #creating a list with the correct number of clusters
    unique_clusters = np.unique(labels_k_sklearn) #getting the unqiue labels (labels for the clusters we do have)

    # if the cluster does not have a unique label, it was not created during clustering, we need to assign it a -1
    for cluster in expected_clusters:
        if cluster not in unique_clusters:
            k_means_sklearn_confusion_matrix = np.insert(k_means_sklearn_confusion_matrix, cluster, -1, axis=1)  # Assign the missing cluster to be -1

print("\nK means confusion matrix using sklearn: \n", k_means_sklearn_confusion_matrix)
"""
K means confusion matrix using sklearn: 
 [177   0   0   0   1   0   0   0   0   0]
 [  0  55  25   0   0   1   2   0  99   0]
 [  1   2 151  12   0   0   0   3   8   0]
 [  0   0   1 163   0   5   0   6   8   0]
 [  0   5   0   0 164   1   0   8   3   0]
 [  0   0   0   5   1 175   1   0   0   0]
 [  1   1   0   0   0   0 177   0   2   0]
 [  0   2   0   0   0   0   0 175   2   0]
 [  1   6   4  48   0   7   2   6 100   0]
 [  0  21   0 144   0   5   0   8   2   0]
"""

k_means_accuracy = metrics.fowlkes_mallows_score(target, majority_labels) #using the sklearn method to calculate the index
print("k means accuracy: ", k_means_accuracy)
#k means accuracy using sklearn:  0.6854437852502713


#b. Agglomerative clustering with Ward linkage (sklearn.cluster.AgglomerativeClustering).

aggl_clustering = AgglomerativeClustering(n_clusters= 10,linkage='ward') #using the sklearn function with Ward linkage
labels = aggl_clustering.fit_predict(data) #get the cluster labels
labels_aggl = labels.copy()#get a copy to keep the original labels

#ensure that the clusters are defined by the majority digit in each cluster
majority_digits = np.zeros(k, dtype=int) #set the majority digit to be 0
# loop through the clusters to find the majority digit of each
for cluster in range(k):
    cluster_index = np.where(labels_aggl == cluster)[0] #find the index of the cluster
    cluster_digits = target[cluster_index] #finding the elements of the cluster
    majority_digit = mode(cluster_digits).mode.item() #finding the majority (mode) digit
    majority_digits[cluster] = majority_digit #set the label to be the majority digit

majority_labels = majority_digits[labels_aggl] #create array with the majority labels

aggl_confusion_matrix = confusion_matrix(target, majority_labels) #creating a confusion matrix comparing the actual clusters with the predicted clusters

# If the clustering resulted in less than 10 clusters, fill the missing positions with -1
num_clusters = len(np.unique(labels)) #finding how many unique clusters there are (ie the number of clusters)
if num_clusters < k:
    expected_clusters = list(range(k)) #creating a list with the correct number of clusters
    unique_clusters = np.unique(labels_aggl) #getting the unique labels (labels for the clusters we do have)
#if the cluster does not have a unique label it was not created during clusters --> we need to assign it a -1
    for cluster in expected_clusters:
        if cluster not in unique_clusters:
            aggl_confusion_matrix = np.insert(aggl_confusion_matrix, cluster, -1, axis=1) #assign the missing cluster to be -1

print("Agglomerative clustering confusion matrix: \n", aggl_confusion_matrix)
"""
Agglomerative clustering confusion matrix: 
 [178   0   0   0   0   0   0   0   0   0]
 [  0 155  27   0   0   0   0   0   0   0]
 [  0   0 166   0   0   0   0   1  10   0]
 [  0   0   0 169   0   0   0   1  13   0]
 [  0   0   0   0 178   0   0   3   0   0]
 [  0   0   0   2   0 179   1   0   0   0]
 [  0   0   0   0   0   0 180   0   1   0]
 [  0   0   0   0   0   0   0 179   0   0]
 [  0   3   4   1   0   0   0   1 165   0]
 [  0  20   0 145   0   2   0  11   2   0]
"""
aggl_accuracy = metrics.fowlkes_mallows_score(target, majority_labels)
print("Agglomerative clustering  accuracy: ", aggl_accuracy)
#Agglomerative clustering  accuracy:  0.8321395046705492


#c. Affinity Propagation (sklearn.cluster.AffinityPropagation).
aff_prop = AffinityPropagation()
labels = aff_prop.fit_predict(data) #get the cluster labels
aff_prop_labels = labels.copy() #create a copy of the cluster labels

#ensure that the clusters are defined by the majority digit in each cluster
k = len(np.unique(aff_prop_labels))  # I was having issues with the number of labels being greater than k, so set k to be length of the labels
majority_digits = np.zeros(k, dtype=int) #set the majority digit to be 0
# loop through the clusters to find the majority digit of each
for cluster in range(k):
    cluster_index = np.where(aff_prop_labels == cluster)[0] #find the index of the cluster
    cluster_digits = target[cluster_index] #finding the elements of the cluster
    majority_digit = mode(cluster_digits).mode.item() #finding the majority (mode) digit
    majority_digits[cluster] = majority_digit #set the label to be the majority digit

majority_labels = majority_digits[aff_prop_labels] #create array with the majority labels

aff_prop_confusion_matrix = confusion_matrix(target, majority_labels) #creating a confusion matrix comparing the actual clusters with the predicted clusters

# If the clustering resulted in less than 10 clusters, fill the missing positions with -1
num_clusters = len(np.unique(aff_prop_labels)) #finding how many unique clusters there are (ie the number of clusters)
if num_clusters < k:
    expected_clusters = list(range(k)) #creating a list with the correct number of clusters
    unique_clusters = np.unique(aff_prop_labels) #getting the unqiue labels (labels for the clusters we do have)
    for cluster in expected_clusters:
        if cluster not in unique_clusters:
            aff_prop_confusion_matrix = np.insert(aff_prop_confusion_matrix, cluster, -1, axis=1)

print("Affinity Propagation confusion matrix: \n", aff_prop_confusion_matrix)
"""
Affinity Propagation confusion matrix: 
 [177   0   0   0   1   0   0   0   0   0]
 [  0 182   0   0   0   0   0   0   0   0]
 [  0   0 176   1   0   0   0   0   0   0]
 [  0   0   0 177   0   2   0   0   0   4]
 [  0   4   0   0 171   0   0   4   0   2]
 [  0   0   0   0   0 180   0   0   0   2]
 [  2   1   0   0   0   0 178   0   0   0]
 [  0   0   0   0   0   0   0 178   0   1]
 [  0  16   2   3   0   1   1   1 149   1]
 [  0   1   0   0   0   3   0   1   1 174]
"""

aff_prop_accuracy = metrics.fowlkes_mallows_score(target, majority_labels) #using the sklearn method to calculate the index
print("Affinity Propagation accuracy: ", aff_prop_accuracy)
#Affinity Propagation accuracy:  0.9404776069104355