# Author: Ryan Q
# Email: qiurui96@gmail.com
# Website & Blog: www.ryanqiu.com
# Github: https://github.com/RyanQ96

'''
Data set: generated from a mixture of two Gaussian variables
'''

import numpy as np
import random
import math
#import matplotlib.pyplot as plt
import time
#import seaborn


#Function used to generate data
def generateData(pi_0,mu_0,sigma_0,pi_1, mu_1,sigma_1):
    '''

    :param mu_0: expectation of first Gaussian
    :param sigma_0: variance of first Gaussian
    :param mu_1: expectation of second Gaussian
    :param sigma_1: expectation of second Gaussian
    :return: Mixture of two distribution

    '''
    # Generate 1000 data for each
    length = 1000

    dataFromGaussian0 = np.random.normal(mu_0,math.sqrt(sigma_0),round(length*pi_0))
    dataFromGaussian1 = np.random.normal(mu_1,math.sqrt(sigma_1),round(length*pi_1))

    trainingData = np.append(dataFromGaussian0,dataFromGaussian1)

    random.shuffle(trainingData)

    return trainingData

# Calculate conditional probability of X based on Gaussina parameters mu and sigma
def calcGauss(dataSetArr, mu, sigma):

    # Using Gaussian probability expression to calculate prbability
    result = (1 / (math.sqrt(2 * math.pi * sigma))) * \
             np.exp(-1*(dataSetArr - mu)*(dataSetArr - mu) / (2*sigma))

    return result

def E_step(dataSetArr, pi_0, mu_0, sigma_0, pi_1, mu_1, sigma_1):
    '''

    :param dataSetArr:
    :param pi_0:
    :param mu_0:
    :param sigma_0:
    :param pi_1:
    :param mu_1:
    :param sigma_1:
    :return:
    '''
    gamma_0 = pi_0 * calcGauss(dataSetArr,mu_0,sigma_0)
    gamma_1 = pi_1 * calcGauss(dataSetArr,mu_1,sigma_1)

    sum = gamma_0 + gamma_1

    gamma_0 = gamma_0 / sum
    gamma_1 = gamma_1 / sum

    return gamma_0, gamma_1

def M_step(dataSetArr, mu_0,mu_1, gamma_0, gamma_1):

    mu_0_new = np.dot(gamma_0, dataSetArr) / np.sum(gamma_0)
    mu_1_new = np.dot(gamma_1, dataSetArr) / np.sum(gamma_1)

    sigma_0_new = np.dot(gamma_0, (dataSetArr - mu_0_new)**2) / np.sum(gamma_0)
    sigma_1_new = np.dot(gamma_1, (dataSetArr - mu_1_new)**2) / np.sum(gamma_1)

    pi_0_new = np.sum(gamma_0) / len(gamma_0)
    pi_1_new = np.sum(gamma_1) / len(gamma_1)

    return mu_0_new, mu_1_new, sigma_0_new, sigma_1_new, pi_0_new, pi_1_new


# EM training process
def EM_Train(dataSet, iter = 500):
    '''

    :param dataSet:
    :param iter:
    :return:
    '''

    dataSetArr = np.array(dataSet)

    # Step1: initialize the parameters
    pi_0 = 0.5; mu_0 = 0; sigma_0 = 1
    pi_1 = 0.5; mu_1 = 1; sigma_1 = 1

    step = 0
    while (step < iter):

        # Step2 - 1: E-Step, calculate posterior probability of hidden variable
        gamma_0, gamma_1 = E_step(dataSetArr, pi_0, mu_0, sigma_0, pi_1, mu_1,sigma_1)

        # Step3 - 2: M-Step, maximize the log-likelihood based on gamma(conditional probability)
        mu_0, mu_1, sigma_0, sigma_1, pi_0, pi_1 = M_step(dataSetArr, mu_0,mu_1, gamma_0, gamma_1)

        step += 1
    return pi_0, mu_0, sigma_0, pi_1, mu_1, sigma_1

if __name__ == '__main__':
    start = time.time()

    # Generate training dataset from mixture of two Gaussian distribution
    # mu and sigma are corresponding parameter of Gaussian distribution based on z
    mu_0 = -2; sigma_0 = 0.5**2
    mu_1 = 0.5; sigma_1 = 1
    # pi is priori distribution of hidden vairable z;
    pi_0 = 0.3; pi_1 = 1 - pi_0

    # Use defined function generateData() to get seudo training data
    dataSet = generateData(pi_0,mu_0,sigma_0,pi_1, mu_1,sigma_1)

    #Show joint distribution of whole dataset
    # seaborn.set_style('darkgrid')
    # seaborn.kdeplot(dataSet,shade=True)
    # plt.show()

    pi_0R, mu_0R,sigma_0R,pi_1R,mu_1R,sigma_1R = EM_Train(dataSet)

    print('----------Setting-------------')
    print('pi0:%.1f, mu0:%.1f, sigma0:%.1f, pi1:%.1f, mu1:%.1f, sigma1:%.1f' % (
        pi_0, mu_0, sigma_0, pi_1, mu_1, sigma_1))

    print('----------Result-------------')
    print('pi0:%.1f, mu0:%.1f, sigma0:%.1f, pi1:%.1f, mu1:%.1f, sigma1:%.1f' % (
        pi_0R, mu_0R, sigma_0R, pi_1R, mu_1R, sigma_1R))
    print(time.time() - start)
    # Use defined function generateData() to get seudo training data



