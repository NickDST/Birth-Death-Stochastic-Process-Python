from sympy.interactive import printing
printing.init_printing(use_latex=True)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


from sympy import Eq, solve_linear_system, Matrix
from numpy import linalg
import numpy as np
import sympy as sp
from sympy import *

x, mu, sigma, w, u, v, theta, r, l, k = sp.symbols('x mu sigma w u v theta r lambda, k')


# simulating the birth death process.

# 1a: create a function that would sample from a categorical distribution

# 1b: create the transition matrix PI

# second: sample the holding times from an exponential distribution

PI = np.zeros([100,100])
lambda_birth = 30
lambda_death = 0.5

#filling in the birth probabilities
for i in range(len(PI[0])-1):
    PI[i][i+1] = lambda_birth / (i * lambda_death + lambda_birth)

#filling in the death probabilities
for i in range(1,len(PI[0])):
    PI[i][i-1] = i*lambda_death / (i * lambda_death + lambda_birth)

print(sum(PI[6]))

initial_conditions = np.zeros(100).reshape(100,1)
initial_conditions[0][0] = 1


## fundamental theorem of simulation
def sample_from_categorical(parameters, u):
    payload = 0
    count = 0
    while (u > payload):
        payload += parameters[count]
        count += 1
    return count-1


def exponential_holding_time(current_number):
    s3 = np.random.uniform(0,1,1)
    uniformly_sampled = s3[0]
    l_s = current_number * lambda_death + lambda_birth
    inverse_cdf = - 1 / l_s * log(1 - u)
    exp_eq = inverse_cdf.subs(l, lambda1)
    holding_time = exp_eq.subs(u, uniformly_sampled)

    return holding_time
