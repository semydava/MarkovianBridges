from math import sqrt
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.stats import norm
import random
import numpy as np
import matplotlib.pyplot as plt
def random_walk(n):
    x, y = 0, 0
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = np.arange(n + 1)
    positions = [y]
    directions = ["UP", "DOWN"]
    for i in range(1, n + 1):
        # Randomly select either UP or DOWN
        prob = random.uniform(0, 1)

        # Move the object up or down
        if prob <= 0.5:
            y += 1
        elif prob > 0.5:
            y -= 1
        # Keep track of the positions
        positions.append(y)
    return positions
def discrete_noise():
    prob = random.uniform(0, 1)
    if prob <= 0.5:
        y = 1
    elif prob > 0.5:
        y = -1
    return y
def AR(T, n):
    sigma = 1.  # Standard deviation.
    mu = 0.  # Mean.
    tau = .05  # Time constant.
    dt = T/n
    #dt = .001  # Time step.
    #T = 1.  # Total time.
    #n = int(T / dt)  # Number of time steps.
    t1 = np.linspace(0., T, n)  # Vector of times.
    t2 = t1.tolist()
    t2.reverse()
    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    z = np.zeros(n)
    intersect = False
    x1[0] = 0
    x2[0] = 0.5
    for i in range(n - 1):
        #n1 = discrete_noise()
        x1_el = x1[i] + dt * (-(x1[i] - mu) / tau) + \
            sigma_bis * sqrtdt * np.random.randn()
        #x1_el = x1[i] + dt * (-(x1[i] - mu) / tau) + \
             #sigma_bis * sqrtdt * n1
        x1[i + 1] = x1_el
        #n2 = discrete_noise()
        x2_el = x2[i] + dt * (-(x2[i] - mu) / tau) + \
                    sigma_bis * sqrtdt * np.random.randn()
        #x2_el = x2[i] + dt * (-(x2[i] - mu) / tau) + \
         #sigma_bis * sqrtdt * n2
        x2[i + 1] = x2_el
    X_1 = {}
    X_2 = {}


    for i,t in zip(t1, range(n)):
        X_1[i] = x1[t]

    for i, t in zip(t2, range(n)):
        X_2[i] = x2[t]

    x2 = list(X_2.values())
    x2.reverse()
    for i in range(n):
        if x1[i] == x2[i]:
            intersect = True
            stop_index = i
            stop_value = x1[stop_index]
            stop_time = t1[i]

    if intersect == True:
        #print(t1[min_d_index])
        Z = {}
        for i in t1:
            if i <= t1[stop_index]:
                Z[i] = X_1[i]
            else:
                Z[i] = X_2[i]

        for i in range(n):
            if i <= stop_index:
                z[i] = x1[i]
            else:
                z[i] = x2[n - i]

        z =  list(Z.values())
        fig, ax = plt.subplots(2)
        #fig.suptitle("AR(1) bridge which approximates OU bridge with stopping time " + str(t1[min_d_index]))
        #ax[0].set_xlabel('Time steps')
        ax[0].set_ylabel('States')
        ax[0].set_xlim(0, T)
        ax[1].set_xlim(0, T)
        ax[1].set_xlabel('Time steps')
        ax[1].set_ylabel('States')
        ax[0].title.set_text('Simulation of two AR(1) processes')
        ax[1].title.set_text('Discrete time Markovian Bridge which approximates OU diffusion bridge')
        ax[0].plot(X_1.keys(), X_1.values(), lw=2, color = 'orange', label="forward in time process")
        #ax[0].scatter(X_1.keys(), X_1.values(), lw=2, color = 'orange')
        ax[0].plot(X_2.keys(), X_2.values(), lw=2, color = 'green', label="time-reversed process")
        #ax[0].scatter(X_2.keys(), X_2.values(), lw=2, color = 'green')
        ax[0].plot(t1[stop_index], x1[stop_index], 'ro', label= "first time processes intersect")
        ax[1].plot(t1[:stop_index + 1],z[:stop_index + 1], color = 'orange')
        ax[1].plot(t1[stop_index:],z[stop_index:], color = 'green')
        #ax[1].scatter(t1, z, lw=2, color = 'blue', label="Markovian Bridge")
        ax[1].plot(t1[stop_index], x1[stop_index], 'ro', label= "first time processes intersect")
        ax[0].legend(loc='upper left')
        plt.legend()
        plt.show()

    return intersect

print(AR(1.0, 10))
def Monte_Carlo(sim_n,time, freq):
    count = 0
    j = 0
    while j < sim_n:
        inter = AR(time, freq)
        if inter == True:
            count += 1
        else:
            count += 0
        j += 1
    prob_inter = (count / sim_n)
    return prob_inter

#print(Monte_Carlo(10, 10.0, 4))

def prob_plot():
    #n = [i*0.0001 for i in range(1, 100)]
    T = [i*0.1 for i in range(1, 200)]
    dt = .001
    n = []
    # y = np.tan(np.sqrt(n1))
    # T = [1, 10, 20, 40, 50, 100]
    prob_freq = []
    for i in T:
        n_el = int(i / dt)
        n.append(n_el)
        p = Monte_Carlo(10, i, 4)
        prob_freq.append(p)
    #y = (np.arctan(np.sqrt(T)) / 2) * 1.5
    plt.title("Probability of stopping time detection depending on time frequency")
    plt.xlabel('Number of time steps')
    plt.ylabel('Probability')
    plt.xlim(0,20/dt)
    plt.plot(n, prob_freq, lw=2, color="orange", label="Probability of stopping time detection")
    #plt.plot(n, y)
    plt.show()

print(prob_plot())
def distributions(n):
    n_list = [i for i in range(0, n)]
    white_n = []
    discrete_n = []
    x = np.linspace(-3, 3, n)
    for i,j in zip(n_list, x):
        w_el = (np.pi*0.5) * np.exp(-0.5*((j)/0.5)**2)
        white_n.append(w_el)
        discrete_n.append(discrete_noise())

    plt.plot(n_list, white_n, color = 'blue')
    plt.plot(n_list, discrete_n, color = 'red')
    plt.show()

#print(distributions(100))
