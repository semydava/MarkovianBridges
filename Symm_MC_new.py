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
     #mc = qe.MarkovChain(P, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
     mc = qe.MarkovChain(P, (0, 1, 2, 3))
     #time = list(np.linspace(0, T, num=N))
     #reversed_time = list(np.linspace(0, T, num=N))
     #reversed_time.reverse()
     X_1 = {}
     X_2 = {}
     X_3 = {}
     X_4 = {}
     Z = {}
     Z_1 = {}
     time_steps = [i for i in range(N)]
     reversed_time_steps = [i for i in range(N)]
     reversed_time_steps.reverse()
     if mc.is_irreducible == True and mc.is_aperiodic == True:
          #print('ok')
          X1 = mc.simulate(ts_length=len(time_steps), init=0)
          X2 = mc.simulate(ts_length=len(time_steps), init=3)

          for i in time_steps:
               X_1[i] = X1[i]
               X_3[i] = X2[i]

          for i,t in zip(reversed_time_steps, time_steps):
               X_2[i] = X2[t]
               X_4[i] = X1[t]

          inter = False

          intersections = {}
          intersections_1 = {}
          for i in time_steps:
               if X_1[i] == X_2[i]:
                    intersections[i] = X_1[i]
               elif X_3[i] == X_4[i]:
                    intersections_1[i] = X_3[i]
                    #print(stop_time)
          z1_values = []
          z2_values = []
          z1_1_values = []
          z2_1_values = []
          t1 = []
          t2 = []
          t1_1 = []
          t2_1 = []
          intersections_time = list(intersections.keys())
          stop_time = max(intersections_time)
          intersections_time_1 = list(intersections_1.keys())
          stop_time_1 = min(intersections_time_1)
          for i in time_steps:
               if i <= stop_time:
                    t1.append(i)
                    Z[i] = X_1[i]
                    z1_values.append(Z[i])
               else:
                    t2.append(i)
                    Z[i] = X_2[i]
                    z2_values.append(Z[i])
          for i in time_steps:
               if i <= stop_time_1:
                    t1_1.append(i)
                    Z_1[i] = X_3[i]
                    z1_1_values.append(Z_1[i])
               else:
                    t2_1.append(i)
                    Z_1[i] = X_4[i]
                    z2_1_values.append(Z_1[i])
          #print(Z[stop_index])
          fig, ax = plt.subplots(2)
          #fig.suptitle("Simple Simulation of Markovian Bridge with stopping time " + str(stop_time))
          #ax[0].set_xlabel('Time steps')
          ax[0].set_ylabel('States')
          ax[0].set_xlim(0, N - 1)
          ax[0].set_ylim(-1, 4)
          ax[1].set_xlabel('Time steps')
          ax[1].set_ylabel('States')
          ax[1].set_xlim(0, N - 1)
          ax[1].set_ylim(-1, 4)
          ax[0].scatter(time_steps, X_1.values(), label = "Process Xˆ1 which starts at a",  linestyle='--', color="blue")
          ax[0].plot(time_steps, X_1.values(), linestyle='--', color="blue")
          #ax[0, 1].scatter(time, X_3.values(), label="Process starts at b at time 0", linestyle='--', color="orange")
          #ax[0, 1].plot(time, X_3.values(), linestyle='--', color="orange")
          ax[0].scatter(reversed_time_steps, X_2.values(), label = "Process Xˆ2 which starts at b", linestyle='--', color="orange")
          ax[0].plot(reversed_time_steps, X_2.values(), linestyle='--', color="orange")
          #ax[0, 1].scatter(reversed_time, X_4.values(), label="Process starts at a at time Δ", linestyle='--',color="blue")
          #ax[0, 1].plot(reversed_time, X_4.values(), linestyle='--', color="blue")
          ax[0].legend(loc='upper left')
          #ax[0, 1].legend(loc='upper left')
          ax[0].plot(stop_time, X1[stop_time], 'ro')
          #ax[0, 1].plot(stop_time_1, stop_value_1, 'ro')
          ax[0].scatter(intersections.keys(), intersections.values(), color = "lawngreen")
          ax[1].plot(time_steps, Z.values(), label= "(0, a, Δ, b) - Markovian bridge", color="fuchsia")
          ax[1].scatter(t1, z1_values, color='blue')
          ax[1].scatter(t2, z2_values, color='orange')
          ax[1].plot(stop_time, X1[stop_time], 'ro')
          #ax[1, 1].plot(time, Z_1.values(), label="(0, b, Δ, a) - Markovian bridge", color="purple")
          #ax[1, 1].scatter(t1_1, z1_1.values(), color="blue")
          #ax[1, 1].scatter(t2_1, z2_1.values(), color="orange")
          #ax[1, 1].plot(stop_time_1, stop_value_1, 'ro')
          ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.)
          #ax[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       #ncol=2, mode="expand", borderaxespad=0.)
          ax[1].legend(loc='upper left')
          fig.savefig('static_plot.png')
          plt.show()


     else:
          print("Matrix P doesn't satisfy ergodicity assumption")





'''P = [[0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05],
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
     ]'''
P = [[0.2, 0.1, 0.4, 0.3], [0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.1, 0.2], [0.4, 0.3, 0.2, 0.1]]
def Monte_Carlo(sim_n, freq):
     count = 0
     j = 0
     while j < sim_n:
          inter = MC_Sim(P, 20, freq)
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
     n = [i for i in range(1, 100)]
     n1 = np.arange(0, 100, 5)
     # y = np.tan(np.sqrt(n1))
     #T = [1, 10, 20, 40, 50, 100]
     prob_freq = []
     prob_time = []
     avrg = []
     iterator = 0
     for i in n:
          p = Monte_Carlo(100, i)
          prob_freq.append(p)
          if iterator < len(n) - 1:
               avr = (p + Monte_Carlo(100, n[iterator + 1])) / 2
               avrg.append(avr)
          iterator += 1
     for i in n:
          p = Monte_Carlo_time_interval(100, i)
          prob_time.append(p)

     plt.title("Probability of stopping time detection depending on time frequency" )
     plt.xlabel('Time frequency')
     plt.ylabel('Probability')
     plt.plot(n, prob_freq, lw=2, color = "orange", label = "Probability of stopping time detection")
     # plt.plot(n1, y)
     plt.axis([0, 100, 0, 1.15])

     # ARCTAN:
     y = (np.arctan(np.sqrt(n1)) / 2) * 1.39
     #plt.plot(n1, y, lw=2, color = "red", label = "Arctan function")

     # SMOOTHED:
     ysmoothed = gaussian_filter1d(avrg, sigma=2)
     new_y =  np.insert(ysmoothed,0,0)
     #plt.plot(n, new_y, lw=2, color = "green", label = "Averaged probability")
     plt.legend()

     #ax[1].plot(n, prob_time)
     plt.show()
print(MC_Sim(P, 5, 5))

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






