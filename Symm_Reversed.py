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
     time = list(np.linspace(0, T, num=N))
     reversed_time = list(np.linspace(0, T, num=N))
     reversed_time.reverse()
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
     intersections = {}
     intersections_1 = {}
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

          for i in time_steps:
               #print(X_1[i], X_2[i])
               print(X_3[i], X_4[i])

               if X_1[i] == X_2[i]:
                    #print(X_1[i], X_2[i])
                    intersections[i] = X_1[i]
               elif X_3[i] == X_4[i]:
                    #print(X_3[i], X_4[i])
                    intersections_1[i] = X_3[i]

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
          '''
     
               for i in range(N):
                    if i <= stop_index:
                         Z[i] = X_1[i]
                    else:
                         Z[i] = X_2[i]
               for i in range(N):
                    if i <= stop_index_1:
                         Z_1[i] = X_3[i]
                    else:
                         Z_1[i] = X_4[i]
               
               z1 = {}
               z2 = {}
               t1 = []
               t2 = []
               z1_1 = {}
               z2_1 = {}
               t1_1 = []
               t2_1 = []
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

               for idx in Z_1.keys():
                    if idx <= stop_index_1:
                         z1_1[idx] = Z_1[idx]
                    else:
                         z2_1[idx] = Z_1[idx]
               for i in time:
                    if i <= stop_time_1:
                         t1_1.append(i)
                    else:
                         t2_1.append(i)
               print(len(t1_1), len(z1_1))
               #print(Z[stop_index]) '''


          fig, ax = plt.subplots(2)
          #fig.suptitle("Simple Simulation of Markovian Bridge with stopping time " + str(stop_time))
          ax[0].set_xlabel('Time steps')
          #ax[0].set_ylabel('States')
          ax[0].set_xlim(0, N-1)
          ax[0].set_ylim(-1, 4)
          ax[1].set_xlabel('Time steps')
          ax[1].set_ylabel('States')
          ax[1].set_xlim(0, N-1)
          ax[1].set_ylim(-1, 4)
          #ax[0].plot(time_steps, X1, linestyle='--', color="blue")
          #ax[0].scatter(time_steps, X1, color="blue")
          #ax[0].plot(reversed_time_steps, X2, linestyle='--', color="orange")
          #ax[0].scatter(reversed_time_steps, X2, color= "orange")
          ax[0].scatter(intersections.keys(),intersections.values() , color= "lawngreen")
          ax[0].plot(time_steps, Z.values(), label="(0, a, Δ, b) - Markovian bridge", color="fuchsia")
          ax[0].plot(stop_time, X1[stop_time], 'ro', label= "last time processes intersect")
          #ax[1].plot(time_steps, X2, linestyle='--', color="orange")
          #ax[1].scatter(time_steps, X2,color= "orange")
          #ax[1].plot(reversed_time_steps, X1, linestyle='--', color="blue")
          #ax[1].scatter(reversed_time_steps, X1, color="blue")
          ax[1].scatter(intersections_1.keys(), intersections_1.values(), color="lawngreen")
          ax[1].plot(time_steps, Z_1.values(), label="(0, b, Δ, a) - Markovian bridge", color="fuchsia")
          ax[1].plot(stop_time_1, X2[stop_time_1], 'ro', label= "first time processes intersect")
          ax[0].scatter(t1, z1_values, color='blue', label = "X^1_t")
          ax[0].scatter(t2,z2_values, color='orange',label = "X^2_{Δ-t}" )
          ax[1].scatter(t1_1, z1_1_values, color='orange', label = "X^2_t")
          ax[1].scatter(t2_1, z2_1_values, color='blue', label = "X^1_{Δ-t}")
          #ax[0].legend(loc='upper left')
          #ax[1].legend(loc='upper left')
          ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=2, mode="expand", borderaxespad=0.)
          ax[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.)
          plt.show()

               #ax[0].scatter(time, X_1.values(), label = "Process Xˆ1 which starts at a",  linestyle='--', color="blue")
               #ax[0].plot(time, X_1.values(), linestyle='--', color="blue")
               #ax[1].scatter(time, X_3.values(), label="Process starts at b at time 0", linestyle='--', color="orange")
               #ax[1].plot(time, X_3.values(), linestyle='--', color="orange")
               #ax[0].scatter(reversed_time, X_2.values(), label = "Process Xˆ2 which starts at b", linestyle='--', color="orange")
               #ax[0].plot(reversed_time, X_2.values(), linestyle='--', color="orange")
               #ax[1].scatter(reversed_time, X_4.values(), label="Process starts at a at time Δ", linestyle='--',color="blue")
               #ax[1].plot(reversed_time, X_4.values(), linestyle='--', color="blue")
               #ax[0].legend(loc='upper left')
               #ax[0, 1].legend(loc='upper left')
               #ax[0].plot(stop_time, stop_value, 'ro', label= "last time processes intersect")
               #ax[1].plot(stop_time_1, stop_value_1, 'ro', label= "first time processes intersect")
               #ax[0].scatter(intersections.keys(), intersections.values(), color = "lawngreen")
               #ax[0].plot(time, Z.values(), label= "(0, a, Δ, b) - Markovian bridge", color="purple")
               #plt.scatter(time, Z.values(),  color="purple")
               #ax[0].scatter(t1, z1.values(), color="blue", label = "X^1_t")
               #ax[0].scatter(t2, z2.values(),  color="orange", label = "X^2_{Δ-t}")
               #ax[1].plot(time, Z_1.values(), label="(0, b, Δ, a) - Markovian bridge", color="green")
               #plt.scatter(time, Z_1.values(),  color="green")
               #ax[1].scatter(t1_1, z1_1.values(), color="orange", label = "X^2_t")
               #ax[1].scatter(t2_1, z2_1.values(), color="blue", label = "X^1_{Δ-t}")
               #ax[1].plot(stop_time, stop_value, 'ro')
               #ax[1, 1].plot(stop_time_1, stop_value_1, 'ro')
               #ax[0].legend(loc='upper left')
               #ax[1].legend(loc='upper left')
               #ax[1, 1].legend(loc='upper left')
               #fig.savefig('static_plot.png')

          #else:
               #inter = False
          #return inter
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






