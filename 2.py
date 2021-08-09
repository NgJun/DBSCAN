import numpy as np
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn import metrics, neighbors
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from queue import Queue

def Data_Illustration(X):
    if X.shape[1] == 2:
        plt.plot(X[:, 0], X[:, 1], 'go', markersize = 4)
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


##Data generation
X = make_moons(500, noise= 0.05)[0]
centers = [[1, 1], [-1, -1], [-1, 1]]
X, label = make_blobs(500, centers=centers, cluster_std=0.4, random_state=0)


## Perform DBScan 
class DBScan:
    def __init__(self, data, epsilon = 1, min_points = 5):
        self.df = data
        self.epsilon = epsilon
        self.min_points = min_points
        self.number_of_labels = 0
        self.noise = 0

    def Euclid_dist(self, x, y):
        return np.sqrt((x[0]-y[0])**2 + (x[1]- y[1])**2) #Euclid Distance 

    def fit(self):
        self.df = np.append(self.df, np.array([[-1]*len(X)]).reshape(-1,1), axis=1)  # Df from (500,2) to (500, 3)
        for x in range(len(self.df)):
            # If the currentPoint is already labeled, then skip
            if self.df[x, 2] != -1:
                continue            
            
            # If the currentPoint is not labeled yet, we gona find its neigbor
            current_point = self.df[x, :2]
            new_neighbors = self.rangeQ(current_point)

            # If the currentPoint reaches less than self.min_point within its vicinity, then the currentpoint will be considered as noise
            if len(new_neighbors) < self.min_points: 
                self.df[x, 2] = self.noise
                continue
            
            # If the currentPoints can reach more than self.min_point, then it is valid and will be labeled as a cluster
            self.number_of_labels += 1
            self.df[x, 2] = self.number_of_labels
            
            found_neighbors = new_neighbors
            q = Queue()
            for xx in new_neighbors:
                q.put(xx)

            #while q.empty() == False:
            for current in found_neighbors:
                #current = q.get()
                if self.df[current, 2] == 0:
                    self.df[current, 2] = self.number_of_labels
                
                if self.df[current, 2] != -1:
                    continue


                self.df[current, 2] = self.number_of_labels

                secondary_point = self.df[current, :2]
                secondary_neighbors = self.rangeQ(secondary_point)
                if len(secondary_neighbors) >= self.min_points:
                    for xx in secondary_neighbors:
                        if xx not in found_neighbors:
                            #q.put(xx)
                            found_neighbors.append(xx)

    def rangeQ(self, current_point):
        new_neighbors = []
        for y in range(len(self.df)):
            if self.Euclid_dist(current_point, self.df[y, :2]) <= self.epsilon:
                new_neighbors.append(y)
                 
        return new_neighbors



test = DBScan(X, 0.2, 5)
test.fit()
Data_Illustration(test.df)

end = 1