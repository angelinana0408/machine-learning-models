# =======================================================================
import sys
import math
# =======================================================================
class DataPoints:
    # -------------------------------------------------------------------
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.isAssignedToCluster = False
    # -------------------------------------------------------------------
    def __key(self):
        return (self.label, self.x, self.y)
    # -------------------------------------------------------------------
    def __eq__(self, other):
        return self.__key() == other.__key()
    # -------------------------------------------------------------------
    def __hash__(self):
        return hash(self.__key())
    # -------------------------------------------------------------------
    @staticmethod
    def getMean(clusters, mean):
        # Initialize the mean for each cluster
        # ****************Please Fill Missing Lines Here*****************
        clusterIndex = 0
        for cluster in clusters:    
            mean_x = 0.0
            mean_y = 0.0
            sum_x = 0.0
            sum_y = 0.0
            count = 0
            for point in cluster:
                sum_x += point.x
                sum_y += point.y
                count +=1
            mean_x = sum_x/count
            mean_y = sum_y/count
            mean[clusterIndex][0] = mean_x
            mean[clusterIndex][1] = mean_y
            clusterIndex +=1
        pass
    # -------------------------------------------------------------------
    @staticmethod
    def getStdDeviation(clusters, mean, stddev):
        # Initialize the std for each cluster
        # ****************Please Fill Missing Lines Here*****************
        clusterIndex = 0
        for cluster in clusters:
            stddev_x = 0.0
            stddev_y = 0.0
            sum_x = 0.0
            sum_y = 0.0
            count = 0
            for point in cluster:
                sum_x += math.pow(point.x-mean[clusterIndex][0],2)
                sum_y += math.pow(point.y-mean[clusterIndex][1],2)
                count +=1
            stddev_x = sum_x/count
            stddev_y = sum_y/count
            stddev[clusterIndex][0] = stddev_x
            stddev[clusterIndex][1] = stddev_y
            clusterIndex +=1
        pass
    # -------------------------------------------------------------------
    @staticmethod
    def getCovariance(clusters, mean, stddev, cov):
        # Initialize the cov for each cluster
        # ****************Please Fill Missing Lines Here*****************
        clusterIndex = 0
        cov_xy = []
        for cluster in clusters:
            sum_cov_xy = 0.0
            for point in cluster:
                sum_cov_xy += (point.x-mean[clusterIndex][0])*(point.y-mean[clusterIndex][1])    
            cov_xy.append(sum_cov_xy/ len(cluster))
            clusterIndex +=1
        for index in range(len(clusters)):
            cov[index][0][0] = stddev[index][0]
            cov[index][1][1] = stddev[index][1]
            cov[index][0][1] = cov[index][1][0] = cov_xy[index]
            
        pass
    # -------------------------------------------------------------------
    @staticmethod
    def getNMIMatrix(clusters, noOfLabels):
        nmiMatrix = [[0 for x in range(len(clusters) + 1)] for y in range(noOfLabels + 1)]
        clusterNo = 0
        for cluster in clusters:
            labelCounts = {}
            for point in cluster:
                if not point.label in labelCounts:
                    labelCounts[point.label] = 0
                labelCounts[point.label] += 1
            max = sys.maxint
            labelNo = 0
            labelTotal = 0
            #SORT according to label num
            labelCounts_sorted = sorted(labelCounts.iteritems(), key=lambda (k, v): (v, k), reverse=True)
            for label, val in labelCounts_sorted:
                nmiMatrix[label - 1][clusterNo] = labelCounts[label]
                labelTotal += labelCounts.get(label)
            nmiMatrix[noOfLabels][clusterNo] = labelTotal
            clusterNo += 1
            labelCounts.clear()

        # populate last col
        lastRowCol = 0
        for i in range(len(nmiMatrix) - 1):
            totalRow = 0
            for j in range(len(nmiMatrix[i]) - 1):
                totalRow += nmiMatrix[i][j]
            lastRowCol += totalRow
            nmiMatrix[i][len(clusters)] = totalRow
        nmiMatrix[noOfLabels][len(clusters)] = lastRowCol
        return nmiMatrix
    # -------------------------------------------------------------------
    @staticmethod
    def calcNMI(nmiMatrix):
        # calculate I
        row = len(nmiMatrix)
        col = len(nmiMatrix[0])
        N = nmiMatrix[row - 1][col - 1]
        I = 0.0
        HOmega = 0.0
        HC = 0.0
        for i in range(row - 1):
            for j in range(col - 1):
                denominator = (float(nmiMatrix[i][col - 1]) * nmiMatrix[row - 1][j])
                if denominator == 0.0:
                    continue
                logPart = (float(N) * nmiMatrix[i][j]) / denominator
                if logPart == 0.0:
                    continue
                I += (nmiMatrix[i][j] / float(N)) * math.log(float(logPart))
                logPart1 = nmiMatrix[row - 1][j] / float(N)
                if logPart1 == 0.0:
                    continue
                HC += nmiMatrix[row - 1][j] / float(N) * math.log(float(logPart1))
            if float(N) == 0.0 or float(N) * math.log(nmiMatrix[i][col - 1] / float(N)) == 0.0:
                continue
            HOmega += nmiMatrix[i][col - 1] / float(N) * math.log(nmiMatrix[i][col - 1] / float(N))

        if math.sqrt(HC * HOmega) == 0.0:
            return 0.0
        return I / math.sqrt(HC * HOmega)
    # -------------------------------------------------------------------
    @staticmethod
    def getNoOFLabels(dataSet):
        labels = set()
        for point in dataSet:
            labels.add(point.label)
        return len(labels)
    # -------------------------------------------------------------------
    @staticmethod
    def writeToFile(noise, clusters, fileName):
        # write clusters to file for plotting
        f = open(fileName, 'w')
        for pt in noise:
            f.write(str(pt.x) + "," + str(pt.y) + ",0" + "\n")
        for w in range(len(clusters)):
            print("Cluster " + str(w) + " size :" + str(len(clusters[w])))
            for point in clusters[w]:
                f.write(str(point.x) + "," + str(point.y) + "," + str((w + 1)) + "\n")
        f.close()
# =======================================================================