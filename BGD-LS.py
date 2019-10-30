import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import metrics

## Implemetation of Linear Regression with BGD and Line Search

class BDGWithLS():
    
    def hypothesis(self, theta, X, n):
        h = np.ones((X.shape[0],1))
        theta = theta.reshape(1,n+1)
        for i in range(0,X.shape[0]):
            h[i] = float(np.matmul(theta, X[i]))
        h = h.reshape(X.shape[0])
        return h
    
    def batchGradientDescentWithLineSearch(self, theta, 𝛼, 𝑇, 𝜖, 𝜏, num_iters, h, X, y, n):
        cost = np.ones(num_iters)
        for i in range(0,num_iters):
            theta_try = theta
            for t in range(𝑇): 
                theta_try[0] = theta_try[0] - (𝛼/X.shape[0]) * sum(h - y)
                for j in range(1,n+1):
                    theta_try[j] = theta_try[j] - (𝛼/X.shape[0]) * sum((h-y) * X.transpose()[j])
                h = self.hypothesis(theta, X, n)
                try_cost = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
#                 print(try_cost)
                if cost[i] - try_cost > 𝜖:   # IF Improved objective, break
                    theta = theta_try
                    cost[i]= try_cost
                    break
                else:
                    𝛼 = 𝜏*𝛼         # Backtrack to smaller rate
    
        theta = theta.reshape(1,n+1)
        return theta, cost
    
    
    def FitLinearRegression(self, X, y, 𝛼, 𝑇, 𝜖, 𝜏,num_iters):
        n = X.shape[1]
        one_column = np.ones((X.shape[0],1))
        X = np.concatenate((one_column, X), axis = 1)
        # initializing the parameter vector...
        theta = np.zeros(n+1)
        # hypothesis calculation....
        h = self.hypothesis(theta, X, n)
        # returning the optimized parameters by Gradient Descent...
        theta, cost = self.batchGradientDescentWithLineSearch(theta, 𝛼, 𝑇, 𝜖, 𝜏, num_iters, h, X, y, n)
        return theta, cost
        
        
𝜏 = 0.5     # Learning rate scaler
𝜖 = 0.001  # Tolerance
𝑇 = 5     # Backtrack
𝛼 = 1    # Learning Rate

LR_BGDLS = BGDWithLS()  # Instantiating Regressor object

bgdls_theta, cost1 = LR_BGDLS.FitLinearRegression(bgd_X_train, bgd_Y_train, 𝛼, 𝑇, 𝜖, 𝜏, 100)


# Plotting cost fuction with iterations 

cost1 = list(cost1)
n_iterations = [x for x in range(100)]
plt.plot(n_iterations, cost1)
plt.grid()
plt.xlabel('No. of iterations')
plt.ylabel('Cost')



print("********************************* Comparing Metrics ***********************************\n")

num_iters = 100 # keeping iterations const


print("Best test MSE for normal BGD = 0.34052759039355346 for alpha =0.1 and iters = 100 \n")

print("Scikit-Learn test MSE for normal BGD = 0.3399815867552159\n")


BGDLS = BGDWithLS()
ls_theta, cost_ls = BGDLS.FitLinearRegression(bgd_X_test, bgd_Y_test, 𝛼, 𝑇, 𝜖, 𝜏, num_iters)
bgd_test_pred = bgd_X_test_1.dot(ls_theta.T)
print('''Line Search test MSE with 𝜏 = {} and 𝜖 = {} is {} !'''
      .format(𝜏,𝜖,metrics.mean_squared_error(bgd_Y_test, bgd_test_pred))) 
