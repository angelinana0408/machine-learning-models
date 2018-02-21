# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


# Reads the data from CSV files, converts it into Dataframe and returns x and y dataframes
def getDataframe(filePath):
    dataframe = pd.read_csv(filePath)
    y = dataframe['y']
    x = dataframe.drop('y', axis=1)
    return x, y

# Applies z-score normalization to the dataframe and returns a normalized dataframe
def applyZScore(dataframe): #dataframe size(1000,100), without 1 col at the first col
    normalized_dataframe = dataframe
    ########## Please Fill Missing Lines Here ##########
    data_x = dataframe.values
    featureNum = data_x.shape[1]
    dataNum = data_x.shape[0]
    #print 'featureNum: ', featureNum
    #print 'dataNum: ', dataNum
    for featureIndex in range(0, featureNum):
        dataSum = 0
        sumTemp = 0
        #calculate miu(mean) for this feature
        for dataIndex in range(0, dataNum):
            #sum all the data for the (featureIndex)th feature 
            dataSum = dataSum + data_x[dataIndex][featureIndex]
        meanValue = dataSum / dataNum
        #calculate standard deviation for this feature
        for dataIndexSD in range(0, dataNum):
            sumTemp = sumTemp + (data_x[dataIndexSD][featureIndex] - meanValue) * (data_x[dataIndexSD][featureIndex] - meanValue)
        SD = np.sqrt(sumTemp / (dataNum - 1))
        #put standardalized value back to the dataframe
        for dataIndexPB in range(0, dataNum):
            standardalized_value = (data_x[dataIndexPB][featureIndex] - meanValue) / SD
            normalized_dataframe.set_value(0, 'x'+str(featureIndex+1), standardalized_value) #the 0 is number 0 not str 0
            #print 'dataIndexPB: ', dataIndexPB
            #print 'featureIndex: ', 'x'+str(featureIndex+1)
            #print 'normalized_dataframe size: ', normalized_dataframe.shape
    
    return normalized_dataframe

# train_x and train_y are numpy arrays
# function returns value of beta calculated using (0) the formula beta = (X^T*X)^ -1)*(X^T*Y)
def getBeta(train_x, train_y): #train_x size(1000,101), with 1 col at the first col from now on
    #print 'train_x shape'
    #print train_x.shape
    beta = np.zeros(train_x.shape[1])
    ########## Please Fill Missing Lines Here ##########
    beta = np.linalg.inv(train_x.transpose().dot(train_x)).dot((train_x.transpose().dot(train_y)))
    return beta
    
# train_x and train_y are numpy arrays
# alpha (learning rate) is a scalar
# function returns value of beta calculated using (1) batch gradient descent
def getBetaBatchGradient(train_x, train_y, alpha):
    beta = np.random.rand(train_x.shape[1])
    ########## Please Fill Missing Lines Here ##########
    #flag = 1
    #count = 0
    rowNum = train_x.shape[0]
    derivative = np.copy(beta)
    numIterations = 1000
    middle = 0 #is a value
    #MSE = 100000
    #MSE = np.iinfo(np.int32).max
    #while flag != 0:
    for iterTimes in range(0, numIterations):
        derivative.fill(0)
        MSE_middle = 0 #is a value
        for row_index in range(0, rowNum):     
            row_temp = train_x[row_index,:]
            middleStep = np.subtract(row_temp.transpose().dot(beta), train_y[row_index])
            MSE_middle = MSE_middle + middleStep*middleStep
            derivative = np.add(derivative , row_temp.dot(middleStep))
        beta = np.subtract(beta , np.multiply(alpha, derivative)) 
        #total square error
        #MSE_temp = (MSE_middle*MSE_middle)/rowNum   
        #if MSE_temp > MSE:
            #break 
        #else:
            #MSE = MSE_temp
            
        #if (beta_temp == beta).all():
            #flag = 0
            #break
        #else:
            #flag = 1
            #beta = np.copy(beta_temp)
        #count = count + 1
        #print 'count'
        #print count
    return beta
    
# train_x and train_y are numpy arrays
# alpha (learning rate) is a scalar
# function returns value of beta calculated using (2) stochastic gradient descent
def getBetaStochasticGradient(train_x, train_y, alpha):
    beta = np.random.rand(train_x.shape[1])
    #beta = getBeta(train_x, train_y)
    ########## Please Fill Missing Lines Here ##########
    rowNum = train_x.shape[0]
    numIterations = 1000
    for iterTimes in range(0, numIterations):
        for row_index in range(0, rowNum): 
            row_temp = train_x[row_index,:]
            derivative = np.multiply((np.subtract(train_y[row_index], row_temp.transpose().dot(beta) )),(row_temp))
            beta = np.add(beta, np.multiply(alpha, derivative))
    return beta

# predicted_y and test_y are the predicted and actual y values respectively as numpy arrays
# function prints the mean squared error value for the test dataset
def compute_mse(predicted_y, test_y):
    mse = 100.0
    ########## Please Fill Missing Lines Here ##########
    dataNum = test_y.shape[0]
    SESum = 0
    for i in range(0,dataNum ):
        SESum = SESum + (predicted_y[i] - test_y[i])*(predicted_y[i] - test_y[i])
    mse = SESum / dataNum
    print 'MSE: ', mse
    
# Linear Regression implementation
class LinearRegression(object):
    # Initializes by reading data, setting hyper-parameters, and forming linear model
    # Forms a linear model (learns the parameter) according to type of beta (0 - closed form, 1 - batch gradient, 2 - stochastic gradient)
    # Performs z-score normalization if z_score is 1
    def __init__(self, beta_type, z_score = 0):
        #make alpha smaller
        self.alpha = 0.001/100
        self.beta_type = beta_type
        self.z_score = z_score

        self.train_x, self.train_y = getDataframe('linear-regression-train.csv')
        self.test_x, self.test_y = getDataframe('linear-regression-test.csv')
        
        if(z_score == 1):
            self.train_x = applyZScore(self.train_x)
            self.test_x = applyZScore(self.test_x)
        
        # Prepend columns of 1 for beta 0, train_x's col plus 1 from now on
        self.train_x.insert(0, 'offset', 1) 
        self.test_x.insert(0, 'offset', 1)
        
        self.linearModel()
    
    # Gets the beta according to input
    def linearModel(self):
        if(self.beta_type == 0):
            self.beta = getBeta(self.train_x.values, self.train_y.values)
            
            print 'Beta: '
            print self.beta
        elif(self.beta_type == 1):
            self.beta = getBetaBatchGradient(self.train_x.values, self.train_y.values, self.alpha)
            print 'Beta: '
            print self.beta
        elif(self.beta_type == 2):
            self.beta = getBetaStochasticGradient(self.train_x.values, self.train_y.values, self.alpha)
            print 'Beta: '
            print self.beta
        else:
            print 'Incorrect beta_type! Usage: 0 - closed form solution, 1 - batch gradient descent, 2 - stochastic gradient descent'
            
    # Predicts the y values of all test points
    # Outputs the predicted y values to the text file named "linear-regression-output_betaType_zScore" inside "output" folder
    # Computes MSE
    def predict(self):
        self.predicted_y = self.test_x.values.dot(self.beta)
        np.savetxt('output/linear-regression-output' + '_' + str(self.beta_type) + '_' + str(self.z_score) + '.txt', self.predicted_y)
        compute_mse(self.predicted_y, self.test_y.values)
        
if __name__ == '__main__':
    # Change 1st paramter to 0 for closed form, 1 for batch gradient, 2 for stochastic gradient
    # Add a second paramter with value 1 for z score normalization
    print '------------------------------------------------'
    print 'Closed Form Without Normalization'
    lm = LinearRegression(0)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Batch Gradient Without Normalization'
    lm = LinearRegression(1)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Stochastic Gradient Without Normalization'
    lm = LinearRegression(2)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Closed Form With Normalization'
    lm = LinearRegression(0, 1)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Batch Gradient With Normalization'
    lm = LinearRegression(1, 1)
    lm.predict()
    
    print '------------------------------------------------'
    print 'Stochastic Gradient With Normalization'
    lm = LinearRegression(2, 1)
    lm.predict()
    print '------------------------------------------------'
