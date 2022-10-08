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
def AR(T, a, b, theta, sigma):
    dt = 0.001
    n = int(T/dt)
    #print(n)
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
        n2 = discrete_noise()
        X1[i + 1] = X1[i] - X1[i] * theta * dt + sigma * np.sqrt(dt) * n1
        X2[i + 1] = X2[i] - X2[i] * theta * dt + sigma * np.sqrt(dt) * n2
    X2 = X2[::-1]
    X1 = list(X1)
    X2 = list(X2)
    for index, (x1, x2) in enumerate(zip(X1, X2)):
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

        fig, ax = plt.subplots(2)
        # #ax[0].set_ylim(-1, 3)
        # #ax[1].set_ylim(-1, 3)
        ax[0].set_xlabel('Time steps')
        ax[0].set_ylabel('States')
        ax[1].set_xlabel('Time steps')
        ax[1].set_ylabel('States')
        ax[0].plot(t, X1, label="Process Xˆ1 which starts at a = " + str(a), color="blue")
        # #ax[0].scatter(t, X1, label="Process Xˆ1 which starts at a = - 0.5", color="blue")
        ax[0].plot(t, X2, label="Process Xˆ2 which starts at b = " + str(b), color="orange")
        # #ax[0].scatter(t, X2, label="Process Xˆ2 which starts at b = 1", color="orange")
        ax[0].plot(t[stop_index], X1[stop_index], 'ro', label= "first time processes intersect with θ = " + str(theta) + " and σ = " + str(sigma))
        ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
        ax[1].plot(t1, z1.values(), color="blue")
        ax[1].plot(t2, z2.values(), color="orange")
        ax[1].plot(t[stop_index], X1[stop_index], 'ro', label="first time processes intersect")
        # #plt.plot(t, X1)
        # #plt.plot(t[stop_index], X1[stop_index], 'ro')
        # #plt.plot(t, X2)
        # # plt.grid(True)
        # # plt.xlabel('t')
        # # plt.ylabel('X')
        plt.show()
    return intersect
print(AR(1., 0, 0.5, 5, 1))

def Monte_Carlo_b1(sim_n,T):
    count = 0
    list1 = []
    for i in range(sim_n):
        inter = AR(T, 0, 0.5)
        count += inter
    prob_inter = (count / sim_n)
    return prob_inter
#print(Monte_Carlo_b1(10, 50))

def Monte_Carlo_b2(sim_n,sigma):
    count1 = 0
    count2 = 0
    count3 = 0
    list1 = []
    for i in range(sim_n):
        inter1 = AR(1., 0, 0.5, 0.9, sigma)
        inter2 = AR(1., -0.5, 0.5, 0.9, sigma)
        inter3 = AR(1., -0.5, 1, 0.9, sigma)
        count1 += inter1
        count2 += inter2
        count3 += inter3
    prob_inter1 = (count1 / sim_n)
    prob_inter2 = (count2 / sim_n)
    prob_inter3 = (count3 / sim_n)

    return [prob_inter1, prob_inter2, prob_inter3]
#print(Monte_Carlo_b2(30, 0.5))
def Monte_Carlo_b3(sim_n,T):
    count = 0
    list1 = []
    for i in range(sim_n):
        inter = AR(T, -0.5, 1)
        count += inter
    prob_inter = (count / sim_n)
    return prob_inter

def prob_plot():
    prob_freq1 = []
    prob_freq2 = []
    prob_freq3 = []
    prob_theta1 = []
    prob_theta2 = []
    prob_theta3 = []
    prob_b = []
    i = 0
    T_s = []
    theta_s = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31,  0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41,  0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61,  0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81,  0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.25, 1.27, 1.28, 1.29, 1.3, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5]
    while i <= 5.:
        i += 0.1
        T_s.append(i)
    for i in theta_s:
        p = Monte_Carlo_b2(50, i)
        print(p)
        prob_theta1.append(p[0])
        prob_theta2.append(p[1])
        prob_theta3.append(p[2])

    # for i in T_s:
    #     print('l')
    #     p1 = Monte_Carlo_b1(20, i)
    #     p2 = Monte_Carlo_b2(20, i)
    #     p3 = Monte_Carlo_b3(20, i)
    #     prob_freq1.append(p1)
    #     prob_freq2.append(p2)
    #     prob_freq3.append(p3)
    plt.title("Probability of stopping time detection")
    plt.xlabel('Different time Δ')
    plt.ylabel('Probability')
    plt.plot(theta_s, prob_theta1, color="orange", label="Probability of stopping time detection with a = 0 and b = 0.5")
    plt.plot(theta_s, prob_theta2, color="lawngreen", label="Probability of stopping time detection with a = -0.5 and b = 0.5")
    plt.plot(theta_s, prob_theta3, color="purple", label="Probability of stopping time detection with a = -0.5 and b = 1")
    #plt.plot(T_s, prob_freq1, color="orange", label="Probability of stopping time detection with a = 0 and b = 0.5")
    #plt.plot(T_s, prob_freq2, color="lawngreen", label="Probability of stopping time detection with a = -0.5 and b = 0.5")
    #plt.plot(T_s, prob_freq3, color="purple", label="Probability of stopping time detection with a = -0.5 and b = 1")
    plt.legend(loc='lower right')
    plt.show()
#print(prob_plot())
