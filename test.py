#from cmath import sqrt
from imaplib import Internaldate2tuple
import random
import numpy as np
import math
from math import sqrt
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
        X1[i + 1] = X1[i] - theta * X1[i] * dt + sigma * np.sqrt(dt) * n1
        X2[i + 1] = X2[i] - theta * X2[i] * dt + sigma * np.sqrt(dt) * n2

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
        

        
        #stop_time = t[stop_index]sss
       
        #fig, ax = plt.subplots(2)
        # #ax[0].set_ylim(-1, 3)
        # #ax[1].set_ylim(-1, 3)
        #ax[0].plot(t, X1, label="Process Xˆ1 which starts at a = - 0.5", color="blue")
        #ax[0].plot(t, X2, label="Process Xˆ2 which starts at b = 1", color="orange")
        #ax[0].plot(t[stop_index], X1[stop_index], 'ro', label= "first time processes intersect")
        # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
        #ax[1].plot(t1, z1.values(), color="blue")
        #ax[1].plot(t2, z2.values(), color="orange")
        #ax[1].plot(t[stop_index], X1[stop_index], 'ro', label="first time processes intersect")
        # ax[1].legend(loc='upper left')
        # #plt.plot(t, X1)
        # #plt.plot(t[stop_index], X1[stop_index], 'ro')
        # #plt.plot(t, X2)
        # # plt.grid(True)
        # # plt.xlabel('t')
        # # plt.ylabel('X')
        #plt.show()
    return intersect
#print(AR(1., 0, 0.5, 0.9, 1))

def Monte_Carlo_sigma(sim_n,sigma):
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(sim_n):
        inter1 = AR(5, -0.5, 0.9, 0.2, sigma)
        inter2 = AR(5, -0.5, 0.9, 0.5, sigma)
        inter3 = AR(5, -0.5, 0.5, 0.9, sigma)
        count1 += inter1
        count2 += inter2
        count3 += inter3

    prob_inter1 = (count1 / sim_n)
    prob_inter2 = (count2 / sim_n)
    prob_inter3 = (count3 / sim_n)

    return [prob_inter1, prob_inter2, prob_inter3]  


def Monte_Carlo(sim_n,T, theta, sigma):
    count1 = 0
    count2 = 0
    count3 = 0

    for i in range(sim_n):
        inter1 = AR(T, 0, 0.5, theta, sigma)
        inter2 = AR(T, -0.5, 0.5, theta, sigma)
        inter3 = AR(T, -0.5, 1, theta, sigma)
        count1 += inter1
        count2 += inter2
        count3 += inter3
   

    prob_inter1 = (count1 / sim_n)
    prob_inter2 = (count2 / sim_n)
    prob_inter3 = (count3 / sim_n)

    return [prob_inter1, prob_inter2, prob_inter3] 

def prob_plot_coef():
    prob_freq1 = []
    prob_freq2 = []
    prob_freq3 = []
    prob_theta1 = []
    prob_theta2 = []
    prob_theta3 = []
    prob_b = []
    i = 0
    sigmas = []
    while i <= 10.:
        i += 0.1
        sigmas.append(i)
    for i in sigmas:
        print('l')
        p = Monte_Carlo_sigma(50, i)
        prob_freq1.append(p[0])
        prob_freq2.append(p[1])
        prob_freq3.append(p[2])
    plt.title("Probability of stopping time detection")
    plt.xlabel('Different sigma coefficients')
    plt.ylabel('Probability')
    plt.plot(sigmas, prob_freq1, color="orange", label="Probability of stopping time detection with a = 0 and b = 0.5")
    plt.plot(sigmas, prob_freq2, color="lawngreen", label="Probability of stopping time detection with a = -0.5 and b = 0.5")
    plt.plot(sigmas, prob_freq3, color="purple", label="Probability of stopping time detection with a = -0.5 and b = 1")
    plt.legend(loc='lower right')
    plt.show()
 
#print(prob_plot_coef())

def distance(x1,y1,x2,y2):
     e1 = (x2 - x1)**2 + (y2 - y1)**2
     return sqrt(e1)

def dist_avr(T):
    dis1 = distance(0, 0, 0.5,T)
    dis2 = distance(0, -0.5, 0.5,T)
    dis3 = distance(0, -0.5, 1,T)
    avrg = (dis1 + dis2 + dis3)/3
        
    return avrg




def prob_plot_time(T, theta, sigma):
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
        lam = dist_avr(i)
        #e = 1 - math.exp(- lam * i)
        e = 1 - math.exp((- theta * i * lam)/2)
        prob_e.append(e)
        print('l')
        p = Monte_Carlo(500, i, theta, sigma)
        prob_freq1.append(p[0])
        prob_freq2.append(p[1])
        prob_freq3.append(p[2])
    plt.title("Probability of stopping time detection")
    plt.xlabel('Different time Δ')
    plt.ylabel('Probability')
    plt.plot(T_s, prob_freq1, color="orange", label="Probability of stopping time detection with a1 = 0, b1 = 0.5, θ = " + str(theta) + " and σ = " + str(sigma))
    plt.plot(T_s, prob_freq2, color="lawngreen", label="Probability of stopping time detection with a2 = -0.5, b2 = 0.5,  θ = " + str(theta) + " and σ = " + str(sigma))
    plt.plot(T_s, prob_freq3, color="purple", label="Probability of stopping time detection with a3 = -0.5, b3 = 1, θ = " + str(theta) + " and σ = " + str(sigma))
    #plt.plot(T_s, prob_e, color="black", label="f(x) = 1 - exp(- λ*Δ)")
    plt.legend(loc=(0.5, 0.01))
    plt.show()
print(prob_plot_time(10, 1, 1))
def prob_plot_time1(T, theta, sigma):
    prob_freq1 = []
    prob_freq2 = []
    prob_freq3 = []
    prob_e1 = []
    prob_e2 = []
    prob_e3 = []
    i = 0
    T_s = []
    while i <= T:
        i += 0.05
        T_s.append(i)
    for i in T_s:
        dis1 = distance(0, 0, 0.5,i)
        dis2 = distance(0, -0.5, 0.5,i)
        dis3 = distance(0, -0.5, 1,i)
        #e = 1 - math.exp(- lam * i)
        e1 = 1 - math.exp((- theta * i * dis1)/2)
        prob_e1.append(e1)
        e2 = 1 - math.exp((- theta * i * dis2)/2)
        prob_e2.append(e2)
        e3 = 1 - math.exp((- theta * i * dis3)/2)
        prob_e3.append(e3)
        print('l')
        p = Monte_Carlo(100, i, theta, sigma)
        prob_freq1.append(p[0])
        prob_freq2.append(p[1])
        prob_freq3.append(p[2])
    fig, ax = plt.subplots(3)
    #plt.title("Probability of stopping time detection")
    ax[0].set_xlabel('Different time Δ')
    ax[0].set_ylabel('Probability')
    ax[1].set_xlabel('Different time Δ')
    ax[1].set_ylabel('Probability')
    ax[2].set_xlabel('Different time Δ')
    ax[2].set_ylabel('Probability')
    ax[0].plot(T_s, prob_freq1, color="orange", label="Probability of stopping time detection with a1 = 0, b1 = 0.5, θ = " + str(theta) + " and σ = " + str(sigma))
    ax[1].plot(T_s, prob_freq2, color="lawngreen", label="Probability of stopping time detection with a2 = -0.5, b2 = 0.5,  θ = " + str(theta) + " and σ = " + str(sigma))
    ax[2].plot(T_s, prob_freq3, color="purple", label="Probability of stopping time detection with a3 = -0.5, b3 = 1, θ = " + str(theta) + " and σ = " + str(sigma))
    ax[0].plot(T_s, prob_e1, color="black", label="f(x) = 1 - exp(- λ*Δ/2)")
    ax[1].plot(T_s, prob_e2, color="black", label="f(x) = 1 - exp(- λ*Δ/2)")
    ax[2].plot(T_s, prob_e3, color="black", label="f(x) = 1 - exp(- λ*Δ/2)")
    ax[0].legend(loc=(0.5, 0.01))
    ax[1].legend(loc=(0.5, 0.01))
    ax[2].legend(loc=(0.5, 0.01))
    plt.show()
#print(prob_plot_time1(10, 1, 1))
def prob_plot_aprox():
    prob_freq1 = []
    prob_freq2 = []
    prob_freq3 = []
    prob_freq4 = []
    prob_freq5 = []
    prob_freq6 = []
    prob_freq7 = []
    prob_freq8 = []
    prob_freq9 = []
    prob_e = []
    i = 0
    T_s = []
    while i <= 10.:
        i += 0.05
        T_s.append(i)
    for i in T_s:
        lam = dist_avr(i)
        e = 1 - np.exp(- lam * i)
        prob_e.append(e)
        print('l')
        p = Monte_Carlo(300, i)
        prob_freq1.append(p[0])
        prob_freq2.append(p[1])
        prob_freq3.append(p[2])
        prob_freq4.append(p[3])
        prob_freq5.append(p[4])
        prob_freq6.append(p[5])
        prob_freq7.append(p[6])
        prob_freq8.append(p[7])
        prob_freq9.append(p[8])

    fig, ax = plt.subplots(3)
    ax[0].set_xlabel('Different time Δ')
    ax[0].set_ylabel('Probability')
    ax[1].set_xlabel('Different time Δ')
    ax[1].set_ylabel('Probability')
    ax[2].set_xlabel('Different time Δ')
    ax[2].set_ylabel('Probability')
    ax[0].plot(T_s, prob_e, color="black", label="f(x) = 1 - exp(- λ*Δ)")
    ax[0].plot(T_s, prob_freq1, color="orange", label="Probability of stopping time detection with a1 = 0 and b1 = 0.5, theta = 1, sigma = 1")
    ax[0].plot(T_s, prob_freq2, color="lawngreen", label="Probability of stopping time detection with a2 = -0.5 and b2 = 0.5, theta = 1, sigma = 1")
    ax[0].plot(T_s, prob_freq3, color="purple", label="Probability of stopping time detection with a3 = -0.5 and b3 = 1, theta = 1, sigma = 1")
    ax[1].plot(T_s, prob_e, color="black", label="f(x) = 1 - exp(- λ*Δ)")
    ax[1].plot(T_s, prob_freq4, color="orange", label="Probability of stopping time detection with a1 = 0 and b1 = 0.5, theta = 0.7, sigma = 2.5")
    ax[1].plot(T_s, prob_freq5, color="lawngreen", label="Probability of stopping time detection with a2 = -0.5 and b2 = 0.5, theta = 0.7, sigma = 2.5")
    ax[1].plot(T_s, prob_freq6, color="purple", label="Probability of stopping time detection with a3 = -0.5 and b3 = 1, theta = 0.7, sigma = 2.5")
    ax[2].plot(T_s, prob_e, color="black", label="f(x) = 1 - exp(- λ*Δ)")
    ax[2].plot(T_s, prob_freq7, color="orange", label="Probability of stopping time detection with a1 = 0 and b1 = 0.5, theta = 3, sigma = 0.5")
    ax[2].plot(T_s, prob_freq8, color="lawngreen", label="Probability of stopping time detection with a2 = -0.5 and b2 = 0.5, theta = 3, sigma= 0.5")
    ax[2].plot(T_s, prob_freq9, color="purple", label="Probability of stopping time detection with a3 = -0.5 and b3 = 1,  theta = 3, sigma = 0.5")
    ax[0].legend(loc=(0.5, 0.01))
    ax[1].legend(loc=(0.5, 0.01))
    ax[2].legend(loc=(0.5, 0.01))
    plt.show()


# Some updates from PC
