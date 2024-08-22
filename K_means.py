import matplotlib.pyplot as plt
import numpy as np
import math

# a. Implement your own k-means algorithm from the lecture slides using Python. (20 points)
# b. Using the k-means algorithm, cluster the data from the attached file cluster_data.txt.

#read in the data file and return the data as an array
cluster_data= []
with open("cluster_data.txt", "r") as file:
    for line in file: #cleaning the text file so that I look at each line and only look at the 2nd and 3rd column --> the first column was just row numbers which I don't think I need to use
        columns = line.strip().split('\t')
        columns = [float(column) for column in columns]
        selected_columns = [float(columns[1]), float(columns[2])]
        cluster_data.append(selected_columns)
cluster_data_array = np.array(cluster_data)


#function to calculate euclidean distance
def find_euclid_distance(i,j):
    x1, y1 = i
    x2, y2 = j
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) #equation for Euclidean distance

"""
first I want to estimate the number of clusters needed by running a sequential algorithm
implement using BSAS:
- start by taking the first element and forming a first cluster
- each following step we make 1 decision  if it belongs to an existing cluster or belongs in a new cluster
- take element, calculate distance to each cluster
- select the closest cluster
- if the distance is greater than the threshold and # clusters < max # clusters, create a new cluster with the element
- if the closest cluster distance is smaller than the threshold or max number of clusters reached  add element to cluster
"""

#function to calculate what the threshold distance theta should be
#I calculate the Euclidean distances between all the points and use the standard deviation of those distances as theta
#there are other ways of calculating theta which could change the results of BSAS, but based on my readings using the standard deviation seems ok
def calculate_theta(data):
    euclid_distances = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            euclid_distances.append(find_euclid_distance(data[i], data[j])) #calculate the euclidean distance between each pair of points
    return np.std(euclid_distances)

theta = calculate_theta(cluster_data_array) #calculated theta based on the values in the data

def BSAS(data, theta, q): #takes in the data in an array, theta which is the distance threshold, and q which is the max number of clusters
    clusters = [[data[0]]] #adding the first element into a cluster
    m = 1 #counter for the number of clusters created so far

    for i in range(1, len(data)):
        closest_cluster = None #setting the variables to find the closest cluster for each element in the data array
        closest_distance = float('inf') #setting the closest distance to infinity

        for j in range(len(clusters)):  #calculating the distance from the element to each cluster
            distance = find_euclid_distance(data[i], clusters[j][0]) #I chose to use the first element of the cluster as the point representative, but this could be changed to use a different value

            if distance < closest_distance: #updating the closest distance and cluster if necessary
                closest_distance = distance
                closest_cluster = j

        if closest_distance > theta and m < q: #creating a new cluster
            clusters.append([data[i]])
            m += 1
        else: #adding the element to the closest cluster if the max number of clusters has been reached or the element distance is less than theta
            clusters[closest_cluster].append(data[i])
    return m #returning the number of clusters

#run BSAS multiple times and plot the number of clusters to find the number of clusters to use for k means
q_results = []
m_results = []

for q in range(1, 25): #running with q ranging from 1 to 25
    m = BSAS(cluster_data_array, theta, q)
    q_results.append(q)
    m_results.append(m)

# plotting the max number of clusters against the number of clusters created
plt.plot(q_results, m_results, marker='o', linestyle='-', color='b')
plt.xlabel('q')
plt.ylabel('m')
plt.title('Results of BSAS')
plt.grid(True)
plt.show()


#select k to be where the number of clusters plateaus
#Based on the graph there should be 8 clusters, k=8
k = 8 #determined using BSAS


#K means function
#takes in the data as an array, the number of clusters, k, which was determined using BSAS, and max number of times to run the function (if the function doesn't terminate on its own)
def k_means(data, k, max_iterations):
    centers = data[np.random.choice(range(len(data)), k, replace=False)] #randomly select k initial centers for the data

    for _ in range(max_iterations): #running the following code until max iterations
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

cluster_data_labels = k_means(cluster_data_array, k, 1000) #running my k means function on  cluster_data.txt using k=8 from BSAS
for i in range(k):
    cluster_points = cluster_data_array[cluster_data_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', alpha=0.7) #creating the "Cluster 1" etc labels

#plotting the results based on the assignment instructions
plt.title('K-Means Clustering')
plt.xlabel('Length')
plt.ylabel('Width')
plt.legend(loc='upper right')
plt.show()