import math
import numpy
#-------------------------------------------------------------------
def log(n):
    return math.log(n)
#-------------------------------------------------------------------
def exp(n):
    return math.exp(n)
#-------------------------------------------------------------------
class logistic:
    #******************************************************
    def __init__(self, parameters):
        self.parameters = numpy.transpose(parameters)
         #******************DATA*******************************
        self.x = numpy.array([[1.0, 60.0, 155.0], [1.0, 64.0,135.0], [1.0, 73.0, 170.0]])   
        self.y = numpy.array([0.0, 1.0, 1.0]) #0--Deceased, 1--Alived
    #******************************************************
    ########## Feel Free to Add Helper Functions ##########
    #******************************************************
    def applyZScore(self):
        
        mean = numpy.zeros(self.x.shape[1])
        mean.fill(0.0)
        SD = numpy.zeros(self.x.shape[1])
        SD.fill(0.0)
        
        for index in range(self.x.shape[1]):
            sum0 = 0.0
            sum1 = 0.0
            for i in range(self.x.shape[0]):
                sum0 = sum0 + self.x[i][index]
                
            mean[index] = sum0 / self.x.shape[0]
            for j in range(self.x.shape[0]):
                sum1 = sum1 + (self.x[j][index] - mean[index])*(self.x[j][index] - mean[index])
            SD[index] = math.sqrt(sum1/(self.x.shape[0]))
        for n in range(self.x.shape[1]):
            for m in range(self.x.shape[0]):
                if SD[n] != 0:
                    self.x[m][n] = (self.x[m][n] - mean[n])/SD[n]
        
    def log_likelihood(self):
        self.applyZScore()
        ll = 0.0
        ##################### Please Fill Missing Lines Here #####################
        temp = self.x.transpose().dot(self.parameters)
        
        for i in range(self.x.shape[0]):
            ll = ll + self.y[i]*(temp[i])-log(1+exp(temp[i]))  
        return ll
    #******************************************************
    def gradients(self):
        gradients = []
        ##################### Please Fill Missing Lines Here #####################
        for j in range(self.x.shape[1]):
            temp = 0.0
            for i in range(self.x.shape[0]):
                expVal = (self.parameters).dot(self.x[i])
                p_xi_beta = exp(expVal)/(1+exp(expVal))
                temp = temp + self.x[i][j]*(self.y[i]-p_xi_beta)
            gradients.append(temp)    
           
        return gradients
    #******************************************************
    def iterate(self):
        ##################### Please Fill Missing Lines Here #####################
        self.applyZScore()
        self.parameters = self.parameters - (numpy.linalg.inv(self.hessian())).dot(self.gradients())
        
        return self.parameters
    #******************************************************
    def hessian(self):
        n = len(self.parameters)
        hessian = numpy.zeros((n, n))
        hessian.fill(0.0)
        ##################### Please Fill Missing Lines Here #####################
        for j in range(n):
            for p in range(n):
                temp = 0.0
                for i in range(self.x.shape[0]):
                    expVal = (self.parameters).dot(self.x[i])
                    p_xi_beta = exp(expVal)/(1+exp(expVal))
                    temp = temp + self.x[i][j]*self.x[i][p]*p_xi_beta*(1-p_xi_beta)
                hessian[j][p] = temp*(-1.0)
        return hessian
#-------------------------------------------------------------------
#parameters = numpy.array([0.000025, 0.000025, 0.000025])
parameters = numpy.array([0.25, 0.25, 0.25])
##################### Please Fill Missing Lines Here #####################
## initialize parameters
l = logistic(parameters)
parameters = l.iterate()
l = logistic(parameters)
print l.iterate()
