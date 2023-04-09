# %%
import pandas as pd                                                 # pandas used only to read the initial table
import numpy as np                                                  # numpy used only for linear algebra operations                              

# %%
df = pd.read_csv('ipl.csv')                                         # read the csv file                  
#print(df.head())                                                          # uncomment to see the first 5 rows of the table                                                                                       

# %%
#print(df.columns)                                                         # uncomment to check the columns of the table                 

# %%
# Preprocessing:
# Step 1:  converting '-' values to 0
# Step 2: droppng player names as it is irrelevant for clustering, and also dropping the target variable 'y' and 'BBI' as they have 0 values for all the players
# Step 3: normalizing data using z-score normalization
df_names = df['PLAYER']
df_replace = df.drop('PLAYER', axis=1)
df_replace = df_replace.drop( ['y', 'BBI'], axis=1)
df_replace.replace('-', 0, inplace=True)
df_replace = df_replace.apply(pd.to_numeric, errors='coerce')                 # convert all the values to numeric to normalize the data   
df_replace = (df_replace - df_replace.mean()) / df_replace.std()
df_replace.replace(np.nan, 0, inplace=True)
X = df_replace.to_numpy()

# %%
# Implementaion of K-Means Clustering as a class

class MyKMeans(object):
    '''
    Class variables:
    k: number of clusters
    iterations: number of iterations to run the algorithm, default set to 20
    X: input data
    centroids: centroids of the clusters
    clusters: list of lists, each list contains the indices of the points in the cluster
    clustered_points: list of lists, each list contains the points in the cluster
    cluster_indices: list of integers, each integer represents the cluster index of the corresponding point in X

    Functions:
    __init__: initializes the class variables
    cosine_similarity_distance: calculates the cosine similarity distance between two vectors
    fit: applies k-means clustering on the input data X
    silhouette_score: evaluates the clustering using the silhouette score metric after the clustering has been done using fit()
    '''

    def __init__(self, k, iterations = 20):


        '''
        Initialize the class variables
        '''

        self.iterations = iterations
        self.k = k

    def cosine_similarity_distance(self, a, b):
        '''
        Returns the cosine similarity distance between two vectors a and b
        '''

        return 1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    def fit(self, X):
        '''
        Applies k-means clustering on the input data X.

        k centroids are randomly chosen from the data. A number of iterations are run until the maximum number of iterations
        is reached or the centroids do not change. In each iteration, point is moved to the cluster with the closest centroid,
        calculated using cosine similarity distance. The centroid of each cluster is then recalculated by taking the mean of
        all the points in each cluster.

        The data is then once again used using the final centroids to give the final clusters and their indices, which are
        stored in the class variables and also returned by the function as (clusters, cluster_indices)
        '''

        self.X = X
        self.centroids = {}
        self.cluster_indices = []
        self.clusters = []
        self.clustered_points = []
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]
        #print(self.centroids)
        last_centroids = np.zeros(self.centroids.shape)
        iters = 0
        while np.not_equal(self.centroids, last_centroids).any() and iters < self.iterations:
            clustered_points = [[] for i in range(self.k)]
            for point in X:
                distances = [self.cosine_similarity_distance(point, centroid) for centroid in self.centroids]
                #print(distances)
                clustered_points[np.argmin(distances)].append(point)                        # add the point to the cluster with the closest centroid
            last_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in clustered_points]
            for i in range(self.k):
                if len(clustered_points[i]) == 0:                                           # if a cluster is empty, set the centroid to the last centroid
                    self.centroids[i] = last_centroids[i]
            iters += 1
        clustered_points = [[] for i in range(self.k)]
        cluster_indices = []
        for point in X:
            distances = [self.cosine_similarity_distance(point, centroid) for centroid in self.centroids]
            clustered_points[np.argmin(distances)].append(point)
            cluster_indices.append(np.argmin(distances))
        self.clusters = [[] for i in range(self.k)]
        for i in range(len(cluster_indices)):
            self.clusters[cluster_indices[i]].append(i)                                     # store the indices of the points in the clusters
        self.clustered_points = clustered_points
        self.cluster_indices = cluster_indices
        return self.clusters, cluster_indices
    
    def silhouette_score(self):
        '''
        Returns the clustering using the silhouette score metric after the clustering has been done using fit().

        List a is evaluated as the average distance between a point and all the other points in the same cluster,
        taking care of the null case by using mean as 0. List b is evaluated as the average distance between a point
        and all the other points in the nearest cluster, which is decided by finding cluster with minimum average 
        distance to the point.
         
        The silhouette score list s_arr is then calculated as (b - a)/max(a, b) for each point and the final score
        s is the mean of all the values in s_arr.
        '''

        a = []
        b = []
        for i in range(len(self.X)):
            ai = np.array([self.cosine_similarity_distance(self.X[i], self.X[j]) for j in range(len(self.X)) if i != j and self.cluster_indices[i] == self.cluster_indices[j]])
            a.append(np.mean(ai) if len(ai) > 0 else 0)
            if np.isnan(a[-1]):
                a[-1] = 1e9
    
        for i in range(len(self.X)):
            bi = []
            for n in range(self.k):
                if n == self.cluster_indices[i]:
                    continue
                temp = [self.cosine_similarity_distance(self.X[i], self.X[j]) for j in range(len(self.X)) if i != j and self.cluster_indices[j] == n]
                if len(temp) == 0:
                    temp = [1e9]
                bi.append(np.mean(temp))
            b.append(min(bi))
            if np.isnan(b[-1]):
                b[-1] = 1e9
        
        s_arr = [(b[i] - a[i]) / (max(a[i], b[i])) for i in range(len(self.X))]
        s = np.mean(s_arr)
        return s
    
    def save_file(self, file_name):
        '''
        Saves the clusters in a file with the name file_name.
        
        Each individual cluster is sorted and then the clusters are sorted according to the first element of each cluster
        (which gives the minimum indexed point in the cluster). These clusters are then saved in the file, with each cluster
        on a new line and all the cluster indices in the cluster separated by a comma.
        '''

        temp_clusters = [sorted(self.clusters[i]) for i in range(len(self.clusters))]
        min_vals = [[temp_clusters[i][0], i] for i in range(len(temp_clusters))]         # stores minimum value of each cluster and its index in the original list
        min_vals = sorted(min_vals, key=lambda x: x[0])
        final_clusters = []
        for i in range(len(temp_clusters)):
            final_clusters.append(temp_clusters[min_vals[i][1]])
        with open(file_name, 'w') as f:
            for i in range(len(final_clusters)):
                for j in range(len(final_clusters[i])):
                    f.write(str(final_clusters[i][j]) + ', ')
                f.write('\n')
        f.close()

        



# %%
# Finding best value of k using KMeans

best_k = -1
best_score_kmeans = -1
best_model_kmeans = None
for k in range(3, 7):
   
    kmeans = MyKMeans(k)
    kmeans.fit(X)
    score = kmeans.silhouette_score()
    print('Silhouette score for k = {} is {}'.format(k, score))
    if score > best_score_kmeans:
        best_score_kmeans = score
        best_model_kmeans = kmeans
        best_k = k
    
print('Model with best silhouette score for k = {} has score {}'.format(best_k, best_score_kmeans))
    

# testing implemntation of silhouette score
# from sklearn.metrics import silhouette_score
# km = MyKMeans(3)
# km.fit(X)
# km.evaluate(X)
# print(silhouette_score(X, km.cluster_indices))
# km.silhouette_score()

# %%
# Implemenation of Single Linkage Top Down Clustering

class singleLinkageTopDown(object):
    '''
    Class variables:
    X: Input data
    k: The number of clusters
    clusters: list of clusters, each cluster is a list of indices of the points in the cluster
    distances: matrix of pairwise distances between all the points in the input data, calculated using cosine similarity
    cluster_indices: list of cluster indices of each point in the input data

    Functions:
    __init__: initializes the class variables
    cosine_similarity_distance: calculates the cosine similarity distance between two vectors
    cluster_diameter: calculates the diameter of a cluster (the maximum distance between two points in the cluster)
    single_linkage: calculates the single linkage distance between two clusters
    complete_linkage: calculates the complete linkage distance between two clusters
    split_point: Finds the point about which a cluster is to be split
    split_cluster: Splits a cluster into two clusters using the extra point
    fit: Performs the clustering
    silhouette_score: Evaluates the clustering using the silhouette score metric
    save_file: Saves the clusters in a file 
    '''

    def __init__(self, k):
        '''
        Initializes the class variable k (number of clusters).
        '''

        self.k = k
        self.X = None
        self.clusters = None
        self.distances = None
        self.cluster_indices = None

    def cosine_similarity_distance(self, a, b):
        '''
        Returns the cosine similarity distance between two vectors a and b
        '''

        return 1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    
    def cluster_diameter(self, cluster):
        '''
        Returns the diameter of cluster.
        
        The diameter of a cluster is the maximum distance between any two points in the cluster.
        '''

        max_dist = 0
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                max_dist = max(max_dist, self.distances[cluster[i]][cluster[j]])
        return max_dist
    
    def single_linkage(self, cluster1, cluster2):
        '''
        Returns min_dist, the single linkage distance between two clusters cluster1 and cluster2.
        
        The single linkage distance between two clusters is the minimum distance between any two points in the two clusters.
        '''

        min_dist = 1e9
        for i in cluster1:
            for j in cluster2:
                min_dist = min(min_dist, self.distances[i][j])
        return min_dist
    
    def complete_linkage(self, cluster1, cluster2):
        '''Returns max_dist, the complete linkage distance between two clusters cluster1 and cluster2.
        
        The complete linkage distance between two clusters is the maximum distance between any two points in the two clusters.
        '''
        max_dist = 0
        for i in cluster1:
            for j in cluster2:
                max_dist = max(max_dist, self.distances[i][j])
        return max_dist
    
    def split_point(self, cluster):
        '''
        Returns extra_point about which cluster is to be split.
        
        The function choose the point which is most distant from the cluster, which is determined by calculating the average
        distance of each point in the cluster from all the other points in the cluster. The point which has the maximum average
        distance is returned.
        '''

        max_avg_dist = 0
        extra_point = -1
        for i in cluster:
            avg_dist = 0
            for j in cluster:
                avg_dist += self.distances[i][j]
            avg_dist /= len(cluster)
            if avg_dist > max_avg_dist:
                max_avg_dist = avg_dist
                extra_point = i
        return extra_point
    
    def split_cluster(self, cluster, extra_point):
        '''
        Returns two clusters obtained by splitting cluster about extra_point.
        
        The function splits the cluster about the extra_point by calculating the single linkage distance of every point in the
        original cluster other than extra point between the original cluster (without the point) and the new cluster, which
        initially containing only the extra_point. If the point's single linkage distance is lesser with the new cluster, it 
        is added to the new cluster, otherwise it is added back to the original cluster.
        '''

        cluster2 = cluster
        cluster2.remove(extra_point)
        cluster1 = [extra_point]
        for i in cluster:
            if i in cluster1:
                continue
            cluster2.remove(i)
            if self.single_linkage(cluster1, [i]) < self.single_linkage(cluster2, [i]):
                cluster1.append(i)
            else:
                cluster2.append(i)
        return cluster1, cluster2

    def fit(self, X):
        '''
        Returns clusters and cluster_indices after performing the clustering on the input data X.
        
        The function precomputes the pairwise distances between all points in X. It then iteratively splits until each point is
        its own cluster. It does this by choosing the cluster with the largest diameter and with more than one point, finding the 
        point with the largest average dissimilarity and then split the cluster about this point. It then uses complete linkage on
        the clusters to merge the clusters which are the closest among all pairwise clusters until k clusters are obtained.
        '''

        self.X = X
        self.clusters = [[i for i in range(len(X))]]
        self.distances = np.zeros((len(X), len(X)))
        for i in range(len(X)):                                                          # compute distance
            for j in range(i+1, len(X)):
                self.distances[i][j] = self.cosine_similarity_distance(X[i], X[j])
                self.distances[j][i] = self.distances[i][j]
        while len(self.clusters) < len(self.X):                                          # iteration
            cluster_to_split = -1                                                        # find cluster to split
            if len(self.clusters) == 1:
                cluster_to_split = 0
            else:
                max_diameter = 0
                for i in range(len(self.clusters)):
                    if len(self.clusters[i]) == 1:
                        continue
                    if self.cluster_diameter(self.clusters[i]) > max_diameter:
                        max_diameter = self.cluster_diameter(self.clusters[i])
                        cluster_to_split = i
            extra_point = self.split_point(self.clusters[cluster_to_split])              # find point to split the cluster
            #print('Extra point is {}'.format(extra_point))
            if len(self.clusters[cluster_to_split]) == 2:                                # split directly if only two points
                self.clusters[cluster_to_split].remove(extra_point)
                self.clusters.append([extra_point])
                continue
            cluster1, cluster2 = self.split_cluster(self.clusters[cluster_to_split], extra_point)     # split the cluster
            self.clusters[cluster_to_split] = cluster1                                                # update the new clusters
            self.clusters.append(cluster2)                                                  
            
        while len(self.clusters) > self.k:                                                            # create k clusters using complete linkage
            min_dist = 1e9
            cluster1 = -1
            cluster2 = -1
            for i in range(len(self.clusters)):
                for j in range(i+1, len(self.clusters)):
                    if self.complete_linkage(self.clusters[i], self.clusters[j]) < min_dist:
                        min_dist = self.complete_linkage(self.clusters[i], self.clusters[j])
                        cluster1 = i
                        cluster2 = j
            self.clusters[cluster1].extend(self.clusters[cluster2])
            self.clusters.remove(self.clusters[cluster2])
        self.cluster_indices = [0 for i in range(len(X))]
        for i in range(len(self.clusters)):
            for j in self.clusters[i]:
                self.cluster_indices[j] = i
        return self.clusters, self.cluster_indices
    
    def silhouette_score(self):
        '''
        Evaluates the clustering using the silhouette score metric after the clustering has been done using fit().

        List a is evaluated as the average distance between a point and all the other points in the same cluster,
        taking care of the null case by using mean as 0. List b is evaluated as the average distance between a point
        and all the other points in the nearest cluster, which is decided by finding cluster with minimum average 
        distance to the point.
         
        The silhouette score list s_arr is then calculated as (b - a) / max(a, b) for each point and the final score
        s is the mean of all the values in s_arr.
        '''
        a = []
        b = []
        for i in range(len(self.X)):
            ai = np.array([self.cosine_similarity_distance(self.X[i], self.X[j]) for j in range(len(self.X)) if i != j and self.cluster_indices[i] == self.cluster_indices[j]])
            a.append(np.mean(ai) if len(ai) > 0 else 0)
            if np.isnan(a[-1]):
                a[-1] = 1e9
    
        for i in range(len(self.X)):
            bi = []
            for n in range(self.k):
                if n == self.cluster_indices[i]:
                    continue
                temp = [self.cosine_similarity_distance(self.X[i], self.X[j]) for j in range(len(self.X)) if i != j and self.cluster_indices[j] == n]
                if len(temp) == 0:
                    temp = [1e9]
                bi.append(np.mean(temp))
            b.append(min(bi))
            if np.isnan(b[-1]):
                b[-1] = 1e9
        
        s_arr = [(b[i] - a[i]) / (max(a[i], b[i])) for i in range(len(self.X))]
        s = np.mean(s_arr)
        return s
    
    def save_file(self, file_name):  
        '''
        Saves the clusters in a file with the name file_name.
        
        Each individual cluster is sorted and then the clusters are sorted according to the first element of each cluster
        (which gives the minimum indexed point in the cluster). These clusters are then saved in the file, with each cluster
        on a new line and all the cluster indices in the cluster separated by a comma.
        '''

        temp_clusters = [sorted(self.clusters[i]) for i in range(len(self.clusters))]
        min_vals = [[temp_clusters[i][0], i] for i in range(len(temp_clusters))]
        min_vals = sorted(min_vals, key=lambda x: x[0])
        final_clusters = []
        for i in range(len(temp_clusters)):
            final_clusters.append(temp_clusters[min_vals[i][1]])
        with open(file_name, 'w') as f:
            for i in range(len(final_clusters)):
                for j in range(len(final_clusters[i])):
                    f.write(str(final_clusters[i][j]) + ', ')
                f.write('\n')
        f.close()
        

# %%
best_model_hdc = None
best_score_hdc = -1
myHDC = singleLinkageTopDown(best_k)
clusters = myHDC.fit(X)
score = myHDC.silhouette_score()
print('Silhouette score for k = {} is {}'.format(best_k, score))
if score > best_score_hdc:
    best_score_hdc = score
    best_model_hdc = myHDC


# %%
# Get the clusters and the cluster indices from the models
hdc_clusters = best_model_hdc.clusters
hdc_cluster_indices = best_model_hdc.cluster_indices
kmeans_clusters = best_model_kmeans.clusters
kmeans_cluster_indices = best_model_kmeans.cluster_indices

# find pairwise jacard similarity between the clusters
# jacard_matrix[i][j] = jacard similarity between hdc_clusters[i] and kmeans_clusters[j]
jacard_matrix = [[0 for i in range(best_k)] for j in range(best_k)]
for i in range(best_k):
    for j in range(best_k):
        set_1 = set(hdc_clusters[i])
        set_2 = set(kmeans_clusters[j])
        jacard_matrix[i][j] = len(set_1.intersection(set_2)) / len(set_1.union(set_2))


print('Jacard similarity matrix is')
for i in range(best_k):
    print(jacard_matrix[i])


# %%
# create mapping between clusters
# mapping[i] = j means hdc_clusters[i] is mapped to kmeans_clusters[j]
# a one-one and onto mapping is created by choosing best jaccard similarity for each kmeans cluster with the hdc cluster
# note that in some cases a kmeans cluster might not get the highest jacard similarity as a mapping as it has been mapped to another cluster already
mapping = {}
for i in range(best_k):
    max_jacard = 0
    max_jacard_index = -1
    possible_indices = [j for j in range(best_k) ]
    for j in range(best_k):
        if j in mapping.keys():
            possible_indices.remove(mapping[j])
    for j in possible_indices:
        if jacard_matrix[i][j] > max_jacard:
            max_jacard = jacard_matrix[i][j]
            max_jacard_index = j
    mapping[i] = max_jacard_index

print('Mapping and Jacard similarities are:')
for i in range(best_k):
    print('{} -> {} with jacard similarity {}'.format(i, mapping[i], jacard_matrix[i][mapping[i]]))



# %%
best_model_kmeans.save_file('kmeans.txt')
best_model_hdc.save_file('divisive.txt')


