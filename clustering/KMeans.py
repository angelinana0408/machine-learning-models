# =======================================================================
from DataPoints import DataPoints
import random
import sys
import math
# =======================================================================
def sqrt(n):
    return math.sqrt(n)
# =======================================================================
class Centroid:
    # -------------------------------------------------------------------
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # -------------------------------------------------------------------
    def __eq__(self, other):
        if not type(other) is type(self):
            return False
        if other is self:
            return True
        if other is None:
            return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        return True
    # -------------------------------------------------------------------
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result
    # -------------------------------------------------------------------
    def toString(self):
        return "Centroid [x=" + self.x + ", y=" + self.y + "]"
    # -------------------------------------------------------------------
    def __str__(self):
        return self.toString()
    # -------------------------------------------------------------------
    def __repr__(self):
        return self.toString()
# =======================================================================
class KMeans:
    # -------------------------------------------------------------------
    def __init__(self):
        self.K = 0
    # -------------------------------------------------------------------
    def main(self, args):
        seed = 71
        
        dataSet = self.readDataSet("dataset1.txt")
        self.K = DataPoints.getNoOFLabels(dataSet)
        random.Random(seed).shuffle(dataSet)
        self.kmeans(dataSet,1)
        
        print("")
        dataSet = self.readDataSet("dataset2.txt")
        self.K = DataPoints.getNoOFLabels(dataSet)
        random.Random(seed).shuffle(dataSet)
        self.kmeans(dataSet,2)
        
        print("")
        dataSet = self.readDataSet("dataset3.txt")
        self.K = DataPoints.getNoOFLabels(dataSet)
        random.Random(seed).shuffle(dataSet)
        self.kmeans(dataSet,3)
    # -------------------------------------------------------------------
    def kmeans(self, dataSet, dataSetID):
        clusters = []
        k = 0
        while k < self.K:
            cluster = set()
            clusters.append(cluster)
            k += 1
        
        # Initially randomly assign points to clusters
        i = 0
        for point in dataSet:
            clusters[i % k].add(point)
            i += 1
        
        # calculate centroid for clusters
        centroids = []
        for j in range(self.K):
            centroids.append(self.getCentroid(clusters[j]))
        
        #empty clusters, the old cluster is just used to get a centeriods array, and mission completed.
        for j in range(self.K):
            clusters[j] = set()
        
        self.reassignClusters(dataSet, centroids, clusters)
        
        # continue till converge
        iteration = 0
        while True:
            iteration += 1
            # calculate centroid for clusters
            centroidsNew = []
            for j in range(self.K):
                centroidsNew.append(self.getCentroid(clusters[j]))
            
            isConverge = True
            #as long as one centroid is not stable, keep calculate centroids
            for j in range(self.K):
                if centroidsNew[j] != centroids[j]:
                    isConverge = False
            if isConverge:
                break
            
            for j in range(self.K):
                clusters[j] = set()
            
            self.reassignClusters(dataSet, centroidsNew, clusters)
            #record the recent centroids centroids[], in order to compare with new centroids in the next iter
            for j in range(self.K):
                centroids[j] = centroidsNew[j]
    
        print("Iteration :" + str(iteration))
        # Calculate purity
        maxLabelCluster = []
        for j in range(self.K):
            #maxLabelCluster store the most frequent label's count in each cluster
            maxLabelCluster.append(self.getMaxClusterLabel(clusters[j]))
purity = 0.0
    for j in range(self.K):
        purity += maxLabelCluster[j]
        purity /= len(dataSet)
        print("Purity is :" + str(purity))
        
        noOfLabels = DataPoints.getNoOFLabels(dataSet)
        nmiMatrix = DataPoints.getNMIMatrix(clusters, noOfLabels)
        nmi = DataPoints.calcNMI(nmiMatrix)
        print("NMI :" + str(nmi))
        
        # write clusters to file for plotting
        f = open("Kmeans_"+str(dataSetID)+".csv", "w")
        for w in range(self.K):
            print("Cluster " + str(w) + " size :" + str(len(clusters[w])))
            for point in clusters[w]:
                f.write(str(point.x) + "," + str(point.y) + "," + str(w) + "\n")
        f.close()
    # -------------------------------------------------------------------
    @staticmethod
    def getMaxClusterLabel(cluster):
        labelCounts = {}
        for point in cluster:
            if not point.label in labelCounts:
                labelCounts[point.label] = 0
            labelCounts[point.label] += 1
        max = -sys.maxint - 1
        for label in labelCounts:
            if max < labelCounts[label]:
                max = labelCounts[label]
        return max
    # -------------------------------------------------------------------
    def reassignClusters(self, dataSet, c, clusters):
        # reassign points based on cluster and continue till stable clusters found
        #each point has a dist, dist between this point and all the centroids
        dist = [0.0 for x in range(self.K)]
        for point in dataSet:
            for i in range(self.K):
                dist[i] = self.getEuclideanDist(point.x, point.y, c[i].x, c[i].y)
            
            minIndex = self.getMin(dist)
            # assign point to the closest cluster
            clusters[minIndex].add(point)
    # ****************Please Fill Missing Lines Here*****************
# -------------------------------------------------------------------
def getMin(self, dist):
    min = sys.maxint
        minIndex = -1
        for i in range(len(dist)):
            if dist[i] < min:
                min = dist[i]
                minIndex = i
    return minIndex
# -------------------------------------------------------------------
def getEuclideanDist(self, x1, y1, x2, y2):
    dist = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
        return dist
    # -------------------------------------------------------------------
    def getCentroid(self, cluster):
        # mean of x and mean of y
        cx = 0
        cy = 0
        
        size = len(cluster)
        # ****************Please Fill Missing Lines Here*****************
        x_sum = 0.0
        y_sum = 0.0
        for point in cluster:
            x_sum += point.x
            y_sum += point.y
        
        cx = x_sum/size
        cy = y_sum/size


return Centroid(cx, cy)
    # -------------------------------------------------------------------
    @staticmethod
    def readDataSet(filePath):
        dataSet = []
        with open(filePath) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            points = line.split('\t')
            x = float(points[0])
            y = float(points[1])
            label = int(points[2])
            point = DataPoints(x, y, label)
            dataSet.append(point)
        return dataSet
# =======================================================================
if __name__ == "__main__":
    k = KMeans()
    k.main(None)