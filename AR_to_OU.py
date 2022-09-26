from math import sqrt
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.stats import norm
import random
import numpy as np
import sdepy
import matplotlib.pyplot as plt
# def random_walk(n):
#     x, y = 0, 0
#     # Generate the time points [1, 2, 3, ... , n]
#     timepoints = np.arange(n + 1)
#     positions = [y]
#     directions = ["UP", "DOWN"]
#     for i in range(1, n + 1):
#         # Randomly select either UP or DOWN
#         prob = random.uniform(0, 1)
#
#         # Move the object up or down
#         if prob <= 0.5:
#             y += 1
#         elif prob > 0.5:
#             y -= 1
#         # Keep track of the positions
#         positions.append(y)
#     return positions
def discrete_noise():
    prob = random.uniform(0, 1)
    if prob <= 0.5:
        y = 1
    elif prob > 0.5:
        y = -1
    return y
def OU(T, a, b):
    np.random.seed(123)
    theta = 1
    mu = 1.5
    sigma = 0.3
    dt = 0.001
    #n = int(T/dt)
    t = np.linspace(0, T, 1001)
    dt = t[1] - t[0]
    print(dt)
    X1 = np.zeros(t.shape)
    X2 = np.zeros(t.shape)
    X1[0] = a
    X2[0] = b
    intersect = 0
    for i in range(t.size - 1):
        X1[i + 1] = X1[i] + mu * (theta - X1[i]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        X2[i + 1] = X2[i] + mu * (theta - X2[i]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    X2 = X2[::-1]
    for x1, x2 in zip(X1, X2):
        index1 = np.where(X1 == x1)[0][0]
        #print(index1)
        # print(len(x1))
        if index1 < len(X1) - 1:
            new_index1 = index1 + 1
            n_x1 = X1[new_index1]
            n_x2 = X2[new_index1]
            #print(n_x1, n_x2, x1, x2)
            # print(n_el1, n_el2)
            if (x1 >= x2 and n_x1 <= n_x2) or (x1 <= x2 and n_x1 >= n_x2):
                #print('ok')
                stop_index = index1
                #print(stop_index)
                intersect = 1
                break
    #if intersect == True:
        #plt.plot(t, X1)
        #plt.plot(t[stop_index], X1[stop_index], 'ro')
        #plt.plot(t, X2)
        # plt.grid(True)
        # plt.xlabel('t')
        # plt.ylabel('X')
        #plt.show()
    return intersect
#print(OU(1, 0, 400))
def AR1(T, a, b):
    sigma = 1.  # Standard deviation.
    mu = 0.  # Mean.
    tau = .05  # Time constant.
    dt = .001  # Time step.
    n = int(T / dt)  # Number of time steps.
    t1 = np.linspace(0., T, n)  # Vector of times.
    t2 = t1.tolist()
    t2.reverse()
    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    z = np.zeros(n)
    intersect = False
    x1[0] = a
    x2[0] = b
    for i in range(n - 1):
        n1 = discrete_noise()
        #x1_el = x1[i] + dt * (-(x1[i] - mu) / tau) + \
            #sigma_bis * sqrtdt * np.random.randn()
        x1_el = x1[i] + dt * (-(x1[i] - mu)/tau) + \
             sigma_bis * sqrtdt * n1
        x1[i + 1] = x1_el
        #n2 = discrete_noise()
        #x2_el = x2[i] + dt * (-(x2[i] - mu) / tau) + \
                    #sigma_bis * sqrtdt * np.random.randn()
        x2_el = x2[i] + dt * (-(x2[i] - mu)/tau) + \
         sigma_bis * sqrtdt * n1
        x2[i + 1] = x2_el
    X_1 = {}
    X_2 = {}


    for i,t in zip(t1, range(n)):
        X_1[i] = x1[t]

    for i, t in zip(t2, range(n)):
        X_2[i] = x2[t]

    x2 = list(X_2.values())
    x2.reverse()
    for e1, e2 in zip(x1, x2):
        index1 = np.where(x1 == e1)[0][0]
        #print(len(x1))
        if index1 < len(x1) - 1:
            new_index1 = index1 + 1
            n_el1 = x1[new_index1]
            n_el2 = x2[new_index1]
            #print(n_el1, n_el2)
            if (e1 >= e2 and n_el1 <= n_el2) or (e1 <= e2 and n_el1 >= n_el2):
                stop_index = index1
                intersect = True
                break
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

#print(AR1(1, -400, 400))
def AR(T, a, b):
    sigma = 1.  # Standard deviation.
    mu = 0.  # Mean.
    tau = .05  # Time constant.
    dt = 0.001
    n = int(T/dt)
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
    x1[0] = a
    x2[0] = b
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
    diff = []

    for i,t in zip(t1, range(n)):
        X_1[i] = x1[t]

    for i, t in zip(t2, range(n)):
        X_2[i] = x2[t]

    x2 = list(X_2.values())
    x2.reverse()
    for i in t1:
        d = abs(X_2[i] - X_1[i])
        diff.append(d)
    min_d = min(diff)
    if min_d <= dt:
        intersect = True
    '''
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
        #plt.legend()
        #plt.show() '''

    return intersect
def random_walk(T, a, b):
    x1, y1 = 0, a
    x2, y2 = 0, b
    dt = 0.001
    n = int(T/dt)
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = [i for i in range(n+1)]
    timepoints_steps = [i * dt for i in range(n + 1)]
    reversed_timepoints = [i for i in range(n + 1)]
    reversed_timepoints_steps = [i * dt for i in range(n + 1)]
    reversed_timepoints.reverse()
    reversed_timepoints_steps.reverse()
    positions_1 = [y1]
    positions_2 = [y2]
    Y_1 = {}
    Y_2 = {}
    intersect = 0
    for i in range(0, n):
        n1 = discrete_noise()
        #n2 = discrete_noise()
        el_1 = positions_1[i] + n1
        el_2 = positions_2[i] + n1
        #print(el_1, el_2)
        # Randomly select either UP or DOWN
        #prob = random.uniform(0, 1)
        # Move the object up or down
        #if prob <= 0.5:
            #y1 = 1
            #y2 = 1
        #elif prob > 0.5:
            #y1 = -1
            #y2 = -1
        # Keep track of the positions
        positions_1.append(el_1)
        positions_2.append(el_2)
    positions_2.reverse()
    for e1, e2 in zip(positions_1, positions_2):
        index1 = positions_1.index(e1)
        # print(len(x1))
        if index1 < len(positions_1) - 1:
            new_index1 = index1 + 1
            n_el1 = positions_1[new_index1]
            n_el2 = positions_2[new_index1]
            # print(n_el1, n_el2)
            if (e1 >= e2 and n_el1 <= n_el2) or (e1 <= e2 and n_el1 >= n_el2):
                stop_index = index1
                intersect = 1
                break
    #plt.plot(timepoints_steps, Y_1.values())
    #plt.plot(reversed_timepoints_steps, Y_2.values())
    #plt.show()

    return intersect
def Monte_Carlo_b1(sim_n,time):
    count = 0
    list1 = []
    for i in range(sim_n):
        inter = random_walk(time, 0, 5)
        count += inter
    prob_inter = (count / sim_n)
    return prob_inter
print(Monte_Carlo_b1(10, 100))

def Monte_Carlo_b2(sim_n,time):
    count = 0
    list1 = []
    for i in range(sim_n):
        inter = random_walk(time, 0, 50)
        count += inter
    prob_inter = (count / sim_n)
    return prob_inter

def Monte_Carlo_b3(sim_n,time):
    count = 0
    list1 = []
    for i in range(sim_n):
        inter = random_walk(time, 0, 100)
        count += inter
    prob_inter = (count / sim_n)
    return prob_inter

def Monte_Carlo1(sim_n, b):
    count = 0
    j = 0
    while j < sim_n:
        inter = AR1(1, 10, 0, b)
        if inter == True:
            count += 1
        else:
            count += 0
        j += 1
    prob_inter = (count / sim_n)
    return prob_inter

def prob_plot(T, b):
    #n = [i*0.0001 for i in range(1, 100)]
    #T = [i*0.1 for i in range(1, 200)]
    Ts = [i for i in range(1, T)]
    b_s = [i for i in range(0, b - 1)]
    time_steps = []
    #time_steps = []
    # y = np.tan(np.sqrt(n1))
    # T = [1, 10, 20, 40, 50, 100]
    prob_freq1 = []
    prob_freq2 = []
    prob_freq3 = []
    prob_b = []
    for i in Ts:
        print('l')
        p1 = Monte_Carlo_b1(10, i)
        p2 = Monte_Carlo_b2(10, i)
        p3 = Monte_Carlo_b3(10, i)
        prob_freq1.append(p1)
        prob_freq2.append(p2)
        prob_freq3.append(p3)

    #for i in b_s:
        #p = Monte_Carlo1(10, i)
        #prob_b.append(p)
    #fig, ax = plt.subplots(1)
    plt.title("Probability of stopping time detection")
    plt.xlabel('Different time Δ')
    plt.ylabel('Probability')
    plt.plot(Ts, prob_freq1, color="orange", label="Probability of stopping time detection with b = 1")
    plt.plot(Ts, prob_freq2, color="lawngreen", label="Probability of stopping time detection with b = 50")
    plt.plot(Ts, prob_freq3, color="purple", label="Probability of stopping time detection with b = 100")
    # ax[1].plot(b_s, prob_b, color="green", label="Probability of stopping time detection")
    #ax[1].set_xlabel('Differences between a and b')
    #ax[1].set_ylabel('Probability')
    plt.legend(loc='lower right')
    #ax[1].legend(loc='lower right')
    #ax[2].legend(loc='lower right')
    plt.show()
#print(prob_plot(10, 10))

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
