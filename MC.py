import quantecon as qe
import numpy as np
import random
import sympy
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from matplotlib.animation import FuncAnimation
from random import randint
from quantecon import MarkovChain
from math import sqrt
from scipy.stats import norm

def MC_Sim(P, T, N):
     mc = qe.MarkovChain(P, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
     time = list(np.linspace(0, T, num=N))
     reversed_time = list(np.linspace(0, T, num=N))
     reversed_time.reverse()
     X_1 = {}
     X_2 = {}
     Z = {}
     time_steps = [i for i in range(N)]
     reversed_time_steps = reversed(time_steps)
     if mc.is_irreducible == True and mc.is_aperiodic == True:
          #print('ok')
          X1 = mc.simulate(ts_length=len(time), init=0)
          X2 = mc.simulate(ts_length=len(time), init=4)

          for i in range(N):
               X_1[i] = X1[i]

          for i,t in zip(reversed_time_steps, time_steps):
               X_2[i] = X2[t]

          inter = False
          for i in range(N):
               if X_1[i] == X_2[i]:
                    inter = True
                    stop_index = i
                    stop_value = X_1[stop_index]
                    stop_time = time[i]
                    #print(stop_time)
                    break


          if inter == True:
               for i in range(N):
                    if i <= stop_index:
                         Z[i] = X_1[i]
                    else:
                         Z[i] = X_2[i]
               z1 = {}
               z2 = {}
               t1 = []
               t2 = []

               for idx in Z.keys():
                    if idx <= stop_index:
                         z1[idx] = Z[idx]
                    else:
                         z2[idx] = Z[idx]
               for i in time:
                    if i <= stop_time:
                         t1.append(i)
                    else:
                         t2.append(i)
               '''
               print(stop_time)
               #print(Z[stop_index])
               fig, ax = plt.subplots(2)
               fig.suptitle("Simple Simulation of Markovian Bridge with stopping time " + str(stop_time))
               ax[0].set_xlabel('Time steps')
               ax[0].set_ylabel('States')
               ax[0].set_xlim(0, T + 1)
               ax[0].set_ylim(0, 12)
               ax[1].set_xlabel('Time steps')
               ax[1].set_ylabel('States')
               ax[1].set_xlim(0, T + 1)
               ax[1].set_ylim(0, 12)
               ax[0].scatter(time, X_1.values(), label = "Process starts at a at time 0",  linestyle='--', color="blue")
               ax[0].plot(time, X_1.values(), linestyle='--', color="blue")
               ax[0].scatter(reversed_time, X_2.values(), label = "Process starts at b at time Δ", linestyle='--', color="orange")
               ax[0].plot(reversed_time, X_2.values(), linestyle='--', color="orange")
               ax[0].legend(loc='upper left')
               ax[0].plot(stop_time, stop_value, 'ro')
               ax[1].plot(time, Z.values(), label="Markovian bridge", color="purple")
               ax[1].scatter(t1, z1.values(), color="blue")
               ax[1].scatter(t2, z2.values(),  color="orange")
               ax[1].plot(stop_time, stop_value, 'ro')
               ax[1].legend(loc='upper left')
               fig.savefig('static_plot.png')
               #plt.show()
               '''
          else:
               inter = False
          return inter
     else:
          print("Matrix P doesn't satisfy ergodicity assumption")





P = [[0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05],
     [0.05, 0.05, 0.3, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05],
     [0.4, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05],
     [0.2, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1],
     [0.3, 0.2, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.05, 0.05],
     [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05],
     [0.05, 0.05, 0.3, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.075, 0.025],
     [0.4, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.1, 0.04, 0.06],
     [0.2, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.05],
     [0.3, 0.2, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.05, 0.05],
     [0.2, 0.3, 0.025, 0.025, 0.025, 0.025, 0.1, 0.1, 0.1, 0.05, 0.05]
     ]

def Monte_Carlo(sim_n, freq):
     count = 0
     j = 0
     while j < sim_n:
          inter = MC_Sim(P, 10, freq)
          if inter == True:
               count += 1
          j += 1
     prob_inter = (count / sim_n)
     return prob_inter
def Monte_Carlo_time_interval(sim_n, T):
     count = 0
     j = 0
     while j < sim_n:
          inter = MC_Sim(P, T, 65)
          if inter == True:
               count += 1
          j += 1
     prob_inter = (count / sim_n)
     return prob_inter
def prob_plot():
     # n = [1, 5, 8, 10, 12, 15, 20, 25, 30, 35, 50, 55, 60, 100]
     n = [i for i in range(1, 1000)]
     #n1 = np.arange(0, 100, 5)
     # y = np.tan(np.sqrt(n1))
     #T = [1, 10, 20, 40, 50, 100]
     prob_freq = []
     prob_time = []
     avrg = []
     iterator = 0
     for i in n:
          p = Monte_Carlo(10, i)
          prob_freq.append(p)
          #if iterator < len(n) - 1:
               #avr = (p + Monte_Carlo(100, n[iterator + 1])) / 2
               #avrg.append(avr)
          #iterator += 1
     #for i in n:
          #p = Monte_Carlo_time_interval(100, i)
          #prob_time.append(p)

     plt.title("Probability of stopping time detection depending on time frequency" )
     plt.xlabel('Time frequency')
     plt.ylabel('Probability')
     plt.plot(n, prob_freq, lw=2, color = "orange", label = "Probability of stopping time detection")
     # plt.plot(n1, y)
     plt.axis([0, 100, 0, 1.15])

     # ARCTAN:
     #y = (np.arctan(np.sqrt(n1)) / 2) * 1.39
     #plt.plot(n1, y, lw=2, color = "red", label = "Arctan function")

     # SMOOTHED:
     #ysmoothed = gaussian_filter1d(avrg, sigma=2)
     #new_y =  np.insert(ysmoothed,0,0)
     #plt.plot(n, new_y, lw=2, color = "green", label = "Averaged probability")
     plt.legend()

     #ax[1].plot(n, prob_time)
     plt.show()
print(prob_plot())

#print(Monte_Carlo_time_interval(100, 100))
#print(prob_plot())
'''P = [[0.05, 0.025, 0.025, 0.025, 0.05, 0.025, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025],
     [0.0, 0.0, 0.0, 0.1, 0.15, 0.15, 0.05, 0.05, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1],
     [0.1, 0.1, 0.025, 0.025, 0.025, 0.05, 0.05, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.05, 0.025, 0.05, 0.3],
     [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.0, 0.0, 0.05, 0.025,  0.025, 0.0, 0.05, 0.05,  0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05],
     [0.15, 0.15, 0.05,  0.0, 0.05, 0.0, 0.05, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.0, 0.0],
     [0.025, 0.025, 0.05, 0.0, 0.05, 0.05, 0.05, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.0, 0.05, 0.0],
     [0.025, 0.025, 0.025, 0.025, 0.15, 0.15, 0.05, 0.0, 0.05, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.0, 0.0, 0.0],
     [0.2, 0.2, 0.02, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03],
     [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1],
     [0.15, 0.15, 0.1, 0.1, 0.5, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.0, 0.05, 0.0, 0.05, 0.0],
     [0.1, 0.1,  0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.0, 0.0, 0.0],
     [0.05, 0.025, 0.025, 0.025, 0.05, 0.025, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025],
     [0.0, 0.0, 0.0, 0.1, 0.15, 0.15, 0.05, 0.05, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1],
     [0.1, 0.1, 0.025, 0.025, 0.025, 0.05, 0.05, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.05, 0.025, 0.05, 0.3],
     [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.0, 0.0, 0.05, 0.025, 0.025, 0.0, 0.05, 0.05, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05],
     [0.15, 0.15, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.0, 0.0],
     [0.025, 0.025, 0.05, 0.0, 0.05, 0.05, 0.05, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.0, 0.05, 0.0],
     [0.025, 0.025, 0.025, 0.025, 0.15, 0.15, 0.05, 0.0, 0.05, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.0, 0.0, 0.0],
     [0.2, 0.2, 0.02, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03],
     [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1],
     [0.15, 0.15, 0.1, 0.1, 0.5, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.0, 0.05, 0.0, 0.05, 0.0],
     [0.1, 0.1, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.0, 0.0, 0.0]]'''






