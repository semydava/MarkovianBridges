import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt
import random
def discrete_noise():
    prob = random.uniform(0, 1)
    if prob <= 0.5:
        y = 1
    elif prob > 0.5:
        y = -1
    return y
def geom_bm(T, a, b, mu, sigma):
    dt = 0.001
    n = int(T/dt)
    t = np.linspace(0, T, n)
    X1 = np.zeros(t.shape)
    X2 = np.zeros(t.shape)
    X1[0] = a
    X2[0] = b
    Z = {}
    intersect = 0
    for i in range(t.size - 1):
        n1 = discrete_noise()
        n2 = discrete_noise()
        X1[i + 1] = X1[i] * math.exp((mu - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * n1)
        X2[i + 1] = X2[i] * math.exp((mu - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * n2)

    X2 = X2[::-1]
    X1 = list(X1)
    X2 = list(X2)
    for index, (x1, x2) in enumerate(zip(X1, X2)):
        if index < len(X1) - 1:
            new_index1 = index + 1
            n_x1 = X1[new_index1]
            n_x2 = X2[new_index1]
            # print(n_x1, n_x2)
            if (x1 >= x2 and n_x1 <= n_x2) or (x1 <= x2 and n_x1 >= n_x2):
                # print(x1, x2, n_x1, n_x2)
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

        # fig, ax = plt.subplots(2)
        # # #ax[0].set_ylim(-1, 3)
        # # #ax[1].set_ylim(-1, 3)
        # ax[0].set_xlabel('Time steps')
        # ax[0].set_ylabel('States')
        # ax[1].set_xlabel('Time steps')
        # ax[1].set_ylabel('States')
        # ax[0].plot(t, X1, label="Process Xˆ1 which starts at a = " + str(a), color="blue")
        # # #ax[0].scatter(t, X1, label="Process Xˆ1 which starts at a = - 0.5", color="blue")
        # ax[0].plot(t, X2, label="Process Xˆ2 which starts at b = " + str(b), color="orange")
        # # #ax[0].scatter(t, X2, label="Process Xˆ2 which starts at b = 1", color="orange")
        # ax[0].plot(t[stop_index], X1[stop_index], 'ro',
        #            label="first time processes intersect with μ = " + str(mu) + " and σ = " + str(sigma))
        # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
        # ax[1].plot(t1, z1.values(), color="blue")
        # ax[1].plot(t2, z2.values(), color="orange")
        # ax[1].plot(t[stop_index], X1[stop_index], 'ro', label="first time processes intersect")
        # # #plt.plot(t, X1)
        # # #plt.plot(t[stop_index], X1[stop_index], 'ro')
        # # #plt.plot(t, X2)
        # # # plt.grid(True)
        # # # plt.xlabel('t')
        # # # plt.ylabel('X')
        # plt.show()
    return intersect

#print(geom_bm(1, 1, 3, 1, 1))

def Monte_Carlo(sim_n, T, mu, sigma):
    count1 = 0
    count2 = 0
    count3 = 0

    for i in range(sim_n):
        inter1 = geom_bm(T, 1, 5, mu, sigma)
        inter2 = geom_bm(T, 2, 5, mu, sigma)
        inter3 = geom_bm(T, 3, 7, mu, sigma)
        count1 += inter1
        count2 += inter2
        count3 += inter3

    prob_inter1 = (count1 / sim_n)
    prob_inter2 = (count2 / sim_n)
    prob_inter3 = (count3 / sim_n)

    return [prob_inter1, prob_inter2, prob_inter3]

def prob_plot_time(T, mu, sigma):
    prob_freq1 = []
    prob_freq2 = []
    prob_freq3 = []
    prob_e = []
    i = 0
    T_s = []
    while i <= T:
        i += 0.05
        T_s.append(i)
    for i in T_s:
        #lam = dist_avr(i)
        #e = 1 - math.exp(- lam * i)
        #e = 1 - math.exp(- theta * i)*sigma**2/theta*2
        #prob_e.append(e)
        print('l')
        p = Monte_Carlo(35, i, mu, sigma)
        prob_freq1.append(p[0])
        prob_freq2.append(p[1])
        prob_freq3.append(p[2])
    plt.title("Probability of stopping time detection")
    plt.xlabel('Different time Δ')
    plt.ylabel('Probability')
    plt.plot(T_s, prob_freq1, color="orange", label="Probability of stopping time detection with a1 = 1, b1 = 5, μ = " + str(mu) + " and σ = " + str(sigma))
    plt.plot(T_s, prob_freq2, color="lawngreen", label="Probability of stopping time detection with a2 = 2, b2 = 5,  μ = " + str(mu) + " and σ = " + str(sigma))
    plt.plot(T_s, prob_freq3, color="purple", label="Probability of stopping time detection with a3 = 3, b3 = 7, μ = " + str(mu) + " and σ = " + str(sigma))
    #plt.plot(T_s, prob_e, color="black", label="f(x) = 1 - exp(- λ*Δ)")
    plt.legend(loc=(0.5, 0.01))
    plt.show()
print(prob_plot_time(5, 1, 1))
