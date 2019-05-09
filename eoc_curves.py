import numpy as np
import math

# Defining activation function and their derivatives
def relu(x):
    return x*(x>0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ssoftplus(x):
    return np.log(1 + e^x) - np.log(2)

def tanh(x):
    return np.tanh(x)

def tanh_dash(x):
    return 4 /(np.exp(x) + np.exp(-x))**2

def tanh_2dash(x):
    return  8*(-np.exp(x) + np.exp(-x))/(np.exp(-x) + np.exp(x))**3

def swish(x):
    return x * sigmoid(x)

def swish_dash(x):
    return (sigmoid(x) + x * sigmoid_dash(x))

def elu(x):
    return x *(x>0) + (np.exp(x)-1) * (x<=0)

def elu_dash(x):
    return (x>0) + np.exp(x) * (x<=0)


######################################### Getting the EOC curve ########################################

# We define a function get_eoc that returns triplets (\sigma_b, \sigma_w, q) on the EOC
def get_eoc(act, act_dash, sigma_bs):
    eoc = []
    for sigma in sigma_bs:
        q = 0
        for i in range(200):
            q = sigma**2 + np.mean(act(np.sqrt(q)*z1)**2)/np.mean(act_dash(np.sqrt(q)*z1)**2)
        eoc.append([sigma, 1/np.sqrt(np.mean(act_dash(np.sqrt(q)*z1)**2)), q])
    return np.array(eoc)

# simulate gaussian variables for mean calculations
N = 500000
z1 = np.random.randn(N)
z2 = np.random.randn(N)



activation = tanh
activation_dash = tanh_dash
sigma_b = [0.05, 0.2, 0.5, 1]
eoc = get_eoc(activation, activation_dash, sigma_b)



######################################## Getting the best EOC point ########################################

def beta_q(sigma_bs, act, act_dash, act_sec):
    q_s = get_eoc(act, act_dash, sigma_bs)
    return [(sigma_b, q * np.mean(act_sec(np.sqrt(q)*z1)**2)/ np.mean(act_dash(np.sqrt(q)*z1)**2) / 2) for (sigma_n, q) in zip(sigma_bs, q_s)]





