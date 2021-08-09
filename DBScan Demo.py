import numpy as np
from numpy.core.defchararray import center
from sklearn import cluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics, neighbors
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from queue import Queue

from sklearn.neighbors import kneighbors_graph

def Debug_find_dpoint(Data, point_value, threshold = 0.02):
    """
    This function is for debuging, aim to find the extract coordination of the point based on its aproxomation
    Input: 
        Data: is the ndarray in shape as (n, 2)
        point_value: a pairs of the point' approximated values
    """
    x_ = point_value[0]
    y_ = point_value[1]
    for p in range(len(Data)):
        x = Data[p, 0]
        y = Data[p, 1]
        if np.sqrt( (x-x_)**2 + (y-y_)**2) <= threshold:
            return (p,(x, y))
    return 0

def Debug_data_illustration(X):
    if X.shape[1] == 2:
        plt.plot(X[:, 0], X[:, 1], 'go', markersize = 4, color = 'red')
        plt.axis('equal')
        plt.show()
    if X.shape[1] == 3:
        for label in np.unique(X[:, 2]):
            array = X[ np.where(X[:, 2] == label)]
            color = (np.random.randint(0, 255)/255, np.random.randint(0, 255)/255, np.random.randint(0, 255)/255)
            plt.plot(array[:, 0], array[:, 1], 'go', alpha = 0.8, color = color)
        plt.axis('equal')
        plt.plot()
        plt.show()

def concate_with_SK_label(data, SKoutput):
    klabel = SKoutput.labels_
    a = klabel.reshape(klabel.shape[0], -1)
    b = np.append(data, a, axis=1)
    return b

## Perform DBScan 
class DBScan:
    def __init__(self, data, epsilon = 1, min_points = 5):
        self.dpoints = data
        self.epsilon = epsilon
        self.min_points = min_points
        self.number_of_labels = 0
        self.noise = 0
        self.fit()

    def Euclid_dist(self, x, y):
        return np.sqrt((x[0]-y[0])**2 + (x[1]- y[1])**2) #Euclid Distance 

    def fit(self):
        minus_one_array = np.ones((self.dpoints.shape[0], 1))*-1
        self.dpoints = np.append(self.dpoints, minus_one_array, axis=1)  # Df from (500,2) to (500, 3)
        for idx in range(len(self.dpoints)):

            if idx == 179:
                debug = 1

            if self.dpoints[idx, 2] != -1:
                continue
            
            neighbors_range = self.rangeQ(self.dpoints[idx, :2])
            if len(neighbors_range) <= self.min_points: #When current point cant reach more than "self.min_points", then it is considered as noise
                self.dpoints[idx, 2] = 0
            
            self.number_of_labels += 1
            self.dpoints[idx, 2] = self.number_of_labels

            for neighbor_index in neighbors_range:
                if self.dpoints[neighbor_index, 2] == 0:
                    self.dpoints[neighbor_index, 2] = self.number_of_labels
                
                if self.dpoints[neighbor_index, 2] != -1:
                    continue

                self.dpoints[neighbor_index, 2] = self.number_of_labels
                neighbors_point = self.dpoints[neighbor_index, :2]
                secondary_neighbor_range = self.rangeQ(neighbors_point)
                if len(secondary_neighbor_range) >= self.min_points:
                    for xx in secondary_neighbor_range:
                        if xx not in neighbors_range:
                            neighbors_range.append(xx)



    def rangeQ(self, current_point):
        new_neighbors = []
        for y in range(len(self.dpoints)):
            if self.Euclid_dist(current_point, self.dpoints[y, :2]) <= self.epsilon:
                new_neighbors.append(y)
                 
        return new_neighbors

class Spectral_Cluster:
    def __init__(self, X):
        self.dpoints = X
        self.D = self.Degree_Maxtrix_gen(self.dpoints)
        self.AdjMatrix = self.Adjacent_matrix_gen(self.dpoints)
        self.clusters = 0
        self.Spectral_Clustering()
    
    def Adjacent_matrix_gen(self, matrixA):
        return kneighbors_graph(matrixA, n_neighbors=5).toarray()
    
    def Degree_Maxtrix_gen(self, maxtrixA):
        return np.diag(maxtrixA.sum(axis=1))
    
    def Laplacian_matrix_gen(self, matrixA):
        result = self.Degree_Maxtrix_gen(matrixA) - self.Adjacent_matrix_gen(matrixA)
        return result
    
    def Spectral_Clustering(self):
        L = self.D - self.AdjMatrix
        vals, vecs = np.linalg.eig(L)
        vecs = vecs[:,np.argsort(vals)]
        vals = vals[np.argsort(vals)]
        clusters = vecs[:,1] > 0
        self.clusters = np.array(clusters, dtype=np.uint8)
        self.dpoints = np.append(self.dpoints, self.clusters, axis=1)
        return clusters

if __name__ == "__main__":
    
    ##Data distribution generation
    X = make_moons(500, noise= 0.05)[0]             # two interleaving half circles.
    centers = [[1, 1], [-1, -1], [-1, 1]]           # Gauss 1
    X, label = make_blobs(1500, centers=centers, random_state=0)
    X, varied = make_blobs(n_samples=500, centers=centers, cluster_std=[0.5, 0.6, 0.2] , random_state=0)
    # X, label = make_circles(n_samples=500, noise=0.1, factor=.2) # Circles
    #X = np.loadtxt('data_distribution_1')                         # Special Case
    #X = np.random.rand(500, 2)                                    # Random 
    
    ## DBScan Demonstration
    test = DBScan(X, epsilon= 0.25, min_points=5) # implementation from scratch
    #kmean = KMeans(n_clusters=2, random_state=0).fit(X) # Kmean
    #dbs = DBSCAN(eps = 0.25, min_samples=5).fit(X)      # DBSCAN by sklearn  
    #spc = Spectral_Cluster(X).Spectral_Clustering()

    ## PLot Clustering Result
    Debug_data_illustration(test.dpoints)

    end = 1