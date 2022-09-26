import random
import numpy as np
import sdepy
import matplotlib.pyplot as plt
from decimal import *
def discrete_noise():
    prob = random.uniform(0, 1)
    if prob <= 0.5:
        y = 1
    elif prob > 0.5:
        y = -1
    return y
def AR(T, a, b):
    theta = 0.9
    mu = 0
    sigma = 1
    dt = 0.001
    n = int(T/dt)
    print(n)
    t = np.linspace(0, T, n)
    #dt = T/n
    X1 = np.zeros(t.shape)
    X2 = np.zeros(t.shape)
    X1[0] = a
    X2[0] = b
    Z = {}
    intersect = 0

    for i in range(t.size - 1):
        n1 = discrete_noise()
        #n2 = discrete_noise()
        X1[i + 1] = X1[i] + mu * (theta - X1[i]) * dt + sigma * np.sqrt(dt) * n1
        X2[i + 1] = X2[i] + mu * (theta - X2[i]) * dt + sigma * np.sqrt(dt) * n1
    X2 = X2[::-1]
    X1 = list(X1)
    X2 = list(X2)
    for index, (x1, x2) in enumerate(zip(X1, X2)):
        #print(x1, x2)
        #index1 = np.where(X1 == x1)[0][0]
        #print(index)
        if index < len(X1) - 1:
            new_index1 = index + 1
            n_x1 = X1[new_index1]
            n_x2 = X2[new_index1]
            #print(n_x1, n_x2)
            if (x1 >= x2 and n_x1 <= n_x2) or (x1 <= x2 and n_x1 >= n_x2):
                #print(x1, x2, n_x1, n_x2)
                intersect = 1
                stop_index = new_index1
                break
    if intersect == 1:
        for i in range(t.size - 1):
            if i <= stop_index:
                Z[i] = X1[i]
            else:
                Z[i] = X2[i]
        z1 = {}
        z2 = {}
        t1 = []
        t2 = []
        for idx in Z.keys():
            if idx <= stop_index:
                z1[idx] = X1[idx]
                t1.append(t[idx])
            else:
                z2[idx] = X2[idx]
                t2.append(t[idx])
        # stop_time = t[stop_index]
        # for i in t:
        #     if i <= stop_time:
        #         t1.append(i)
        #     else:
        #         t2.append(i)
        fig, ax = plt.subplots(2)
        #ax[0].set_ylim(-1, 3)
        #ax[1].set_ylim(-1, 3)
        ax[0].plot(t, X1, label="Process Xˆ1 which starts at a = - 0.5", color="blue")
        #ax[0].scatter(t, X1, label="Process Xˆ1 which starts at a = - 0.5", color="blue")
        ax[0].plot(t, X2, label="Process Xˆ2 which starts at b = 1", color="orange")
        #ax[0].scatter(t, X2, label="Process Xˆ2 which starts at b = 1", color="orange")
        ax[0].plot(t[stop_index], X1[stop_index], 'ro', label= "first time processes intersect")
        ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
        ax[1].plot(t1, z1.values(), color="blue")
        ax[1].plot(t2, z2.values(), color="orange")
        ax[1].plot(t[stop_index], X1[stop_index], 'ro', label="first time processes intersect")
        ax[1].legend(loc='upper left')
        #plt.plot(t, X1)
        #plt.plot(t[stop_index], X1[stop_index], 'ro')
        #plt.plot(t, X2)
        # plt.grid(True)
        # plt.xlabel('t')
        # plt.ylabel('X')
        #plt.show()
    return intersect
#print(AR(1., 0, 0.5))

def Monte_Carlo_b1(sim_n,T):
    count = 0
    list1 = []
    for i in range(sim_n):
        inter = AR(T, 0, 0.5)
        count += inter
    prob_inter = (count / sim_n)
    return prob_inter
#print(Monte_Carlo_b1(10, 50))

def Monte_Carlo_b2(sim_n,T):
    count = 0
    list1 = []
    for i in range(sim_n):
        inter = AR(T, -0.5, 0.5)
        count += inter
    prob_inter = (count / sim_n)
    return prob_inter
#print(Monte_Carlo_b2(10, 30000))
def Monte_Carlo_b3(sim_n,T):
    count = 0
    list1 = []
    for i in range(sim_n):
        inter = AR(T, -0.5, 1)
        count += inter
    prob_inter = (count / sim_n)
    return prob_inter

def prob_plot():
    #n_s = [i for i in (range(1, n+1))]
    #T_s = [i*0.01 for i in n_s]
    #T_s = [0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016,  0.0017,  0.0018, 0.0019, 0.002, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.003, 0.0031, 0.0032, 0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039, 0.004, 0.0041, 0.0042, 0.0043, 0.0044, 0.0045, 0.0046, 0.0047, 0.0048, 0.0049, 0.005, 0.0051, 0.0052, 0.0053, 0.0054, 0.0055, 0.0056, 0.0057, 0.0058, 0.0059, 0.006, 0.0061, 0.0062, 0.0063, 0.0064, 0.0065, 0.0066, 0.0067, 0.0068, 0.0069, 0.007, 0.0071, 0.0072, 0.0073, 0.0074, 0.0075, 0.0076, 0.0077, 0.0078, 0.0079, 0.008, 0.0081, 0.0082, 0.0083, 0.0084, 0.0085, 0.0086, 0.0087, 0.0028, 0.0089, 0.009, 0.0091, 0.0092, 0.0093, 0.0094, 0.0095, 0.0096, 0.0097, 0.0098, 0.0099, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016,  0.017,  0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 5.0, 10.0]
    #n_s = [i for i in range(1, T+1)]
    #n_s = [int(i/0.001) for i in range(1, T+1)]
    #n_s = [int(i/0.001) for i in range(1, T+1)]
    prob_freq1 = []
    prob_freq2 = []
    prob_freq3 = []
    prob_b = []
    i = 0
    T_s = []
    while i <= 5.:
        i += 0.1
        T_s.append(i)
    for i in T_s:
        print('l')
        p1 = Monte_Carlo_b1(20, i)
        p2 = Monte_Carlo_b2(20, i)
        p3 = Monte_Carlo_b3(20, i)
        prob_freq1.append(p1)
        prob_freq2.append(p2)
        prob_freq3.append(p3)
    plt.title("Probability of stopping time detection")
    plt.xlabel('Different time Δ')
    plt.ylabel('Probability')
    plt.plot(T_s, prob_freq1, color="orange", label="Probability of stopping time detection with a = 0 and b = 0.5")
    plt.plot(T_s, prob_freq2, color="lawngreen", label="Probability of stopping time detection with a = -0.5 and b = 0.5")
    plt.plot(T_s, prob_freq3, color="purple", label="Probability of stopping time detection with a = -0.5 and b = 1")
    plt.legend(loc='lower right')
    plt.show()
print(prob_plot())
