# coding = utf-8
# This Python program is to calculate
# the NE(Nash Equilibrium) strategy point
# with the Dynamic Programming method, i.e., the analytical result.

# This Program outputs:
# (1) ./RecursiveMth_Output/DiffGame_BVPDynPrgRecur_div{}.npz (the result npz file);
# (2) ./RecursiveMth_Output/outputstr_DiffGame_BVPDynPrgRecur_div{}.txt (the log file).

# note:
# 1.with (D1+D2)/2 as running cost
# 2.with cmd python XX.py (10~16)divs
# 3.with the Pontryagin analysis, the NE strategy should be Min (i.e., 0) or Max Control.
# 4.with the Dynamic Progamming method (Algorithm in Computer Science) to solve min-max control.

# the temporal cost of calcualting the NE strategy.
# T=500 div = 10 0:00:01.988677
# T=500 div = 11 0:00:34.270277
# T=500 div = 12 0:00:01.988677
# T=500 div = 13 0:00:01.988677
# T=500 div = 14 10mins
# T=501 div = 15 3:22:15.801473
# T=501 div = 16 3:00:17.317307
# T=501 div = 17 12:49:12.067169

# import scipy.integrate
# this package can calculate the intergal of function.
import datetime
import numpy as np
import sys
import os
from memory_profiler import profile

# the whole simulation duration
T = 500
# the number of nodes
N = 100
# the frequency of node concacts
lam = 0.004

# computationresult1 sum_U:15.0 sum_rho:1.5
# c = 1.
# alpha = 1.
# beta = 0.1
# computationresult2 sum_U:422.0 sum_rho:32.5
# c = 1.
# alpha = 0.1
# beta = 0.01
# computationresult3 sum_U:404.0 sum_rho:39.5
c = 1
alpha = 1
beta = 0.1

# the Max Control and the Min Control
U_max = 1
U_min = 0
rho_max = 0.5
rho_min = 0

# the initial states of I and D
I0 = 0
D0 = 0
t0 = 0

# Based on the Pontryagin Analysis,
# we can form the candidate choices
# (i.e., four situations: (U_min, rho_min),(U_min, rho_max),(U_max, rho_min),(U_max,rho_max)  )
candidate_choices = []
for i in [U_min, U_max]:
    for j in [rho_min, rho_max]:
        candidate_choices.append((i, j))
# [U_min rho_min][U_min rho_max]
# [U_max rho_min][U_max rho_max]

# based on the control (U and rho) and the state (I and D),
# we can get the derivate of I and D
def get_derivate_ID(tmp_U, tmp_rho, I, D):
    derivate_I = lam*(N-I)-tmp_rho*I/N
    derivate_D = tmp_rho*I/N-lam*D-1/N*D*tmp_U
    return derivate_I, derivate_D

# main alg., to obtain the Nash Equilibrium solution
def V(t, I, D):
    # print('in function V(t,I,D) begin t：{}, I:{}, D:{}'.format(t, I, D))
    if t>=T:
        return 0, []
    # find min max V(t, I, D) with the Comparing these four situations.
    # [four situations: (U_min, rho_min),(U_min, rho_max),(U_max, rho_min),(U_max,rho_max)]
    # tmp value is stored in list_value (Python LIST []).
    list_value = []
    for tunple in candidate_choices:
        tmp_U = tunple[0]
        tmp_rho = tunple[1]

        # based on the current state and the employed controls,
        # we can obtain the new I and the new D
        derivate_I, derivate_D = get_derivate_ID(tmp_U, tmp_rho, I, D)
        I_new = I + h*derivate_I
        D_new = D + h*derivate_D
        # get the Running Cost in the time duration [t, t+h]
        term2 = h * (c * (D+D_new)/2 + alpha * tmp_U - beta * tmp_rho)

        # with the Recursive call method,
        # based on the new state at the time 't+h',
        # we can get the Running Cost in the time duration [t+h, T]
        term1, t_next_behaviors = V(t+h, I_new, D_new)

        # get the Cost at this time (t)
        # Choose the Equilibrium Solution with Comparision Method
        value = term1 + term2
        list_value.append((value, tmp_U, tmp_rho, t_next_behaviors))
    # print(list_value[0][0], list_value[1][0], list_value[2][0], list_value[3][0])

    #  Umin rho_min
    if list_value[0][0] <= list_value[2][0] and list_value[0][0] >= list_value[1][0]:
        # (Umin rhomin) list_value[0]
        t_behaviors_new = list_value[0][3].copy()
        # Insert the current behaviors (U and rho), the current state (I and D), the value to the List.
        t_behaviors_new.insert(0, (t, list_value[0][1], list_value[0][2], I, D, list_value[0][0]))
        return list_value[0][0], t_behaviors_new

    if list_value[1][0] <= list_value[3][0] and list_value[1][0] >= list_value[0][0]:
        # (Umin rhomax) list_value[1]
        t_behaviors_new = list_value[1][3].copy()
        t_behaviors_new.insert(0, (t, list_value[1][1], list_value[1][2], I, D, list_value[1][0]))
        return list_value[1][0], t_behaviors_new

    if list_value[2][0] <= list_value[0][0] and list_value[2][0] >= list_value[3][0]:
        # (Umax rhomin) list_value[2]
        t_behaviors_new = list_value[2][3].copy()
        t_behaviors_new.insert(0, (t, list_value[2][1], list_value[2][2], I, D, list_value[2][0]))
        return list_value[2][0], t_behaviors_new

    if list_value[3][0] <= list_value[1][0] and list_value[3][0] >= list_value[2][0]:
        # (Umax rhomax) list_value[3]
        t_behaviors_new = list_value[3][3].copy()
        t_behaviors_new.insert(0, (t, list_value[3][1], list_value[3][2], I, D, list_value[3][0]))
        return list_value[3][0], t_behaviors_new

# # computation memory
# def mem_main(I0, D0):
#     value, t_behaviors = V(0, I0, D0)
#     return value, t_behaviors
#
# if __name__ == "__main__":
#     div = 10
#     x = np.linspace(0, T, div + 1)
#     h = T / div
#     value, t_behaviors = mem_main(I0, D0)


if __name__ == "__main__":
    # print(sys.argv)
    # div = int(sys.argv[1])

    times = 3
    for tmp_times in range(times):
        computation_time_file = 'ComputationTime_{}times.csv'.format(tmp_times)

        # divide the whole simulation duration into 'div' segments.
        div = 10
        while div <= 13:
            x = np.linspace(0, T, div + 1)
            # get the temporal length of one temporal segment of the whole duration
            h = T / div

            print('T:{} div:{}'.format(T, div))
            print('Umax:{} rhomax:{}'.format(U_max, rho_max))
            outputstr = 'T:{} div:{}\nU_max:{} rho_max:{}\n'.format(T, div, U_max, rho_max)

            t_begin = datetime.datetime.now()
            print('begin at: {}'.format(t_begin))
            # conduct the computation of the NE strategy.
            value, t_behaviors = V(0, I0, D0)
            t_end = datetime.datetime.now()
            print('end at: {}'.format(t_end))
            delta_t = t_end - t_begin
            print(delta_t)
            # record computation time
            ct_file = open(computation_time_file,'a+')
            ct_file.write('{},{},{}\n'.format(div, delta_t.total_seconds(), delta_t))
            ct_file.close()

            outputstr = outputstr+'begin at: {}\n'.format(t_begin)+'end at: {}\n'.format(t_end)+'{}\n'.format(delta_t)

            print(value)
            print(t_behaviors)
            outputstr = outputstr + '{}\n{}\n'.format(value, t_behaviors)

            # Arrange the results into
            statebehaviors = np.zeros((len(t_behaviors), 6), dtype='float')
            for i in range(len(t_behaviors)):
                # time t
                statebehaviors[i][0] = t_behaviors[i][0]
                # U(t), the control
                statebehaviors[i][1] = t_behaviors[i][1]
                # rho(t), the control
                statebehaviors[i][2] = t_behaviors[i][2]
                # I(t), the state
                statebehaviors[i][3] = t_behaviors[i][3]
                # D(t), the state
                statebehaviors[i][4] = t_behaviors[i][4]
                # V(t), value
                statebehaviors[i][5] = t_behaviors[i][5]
            # filename = 'BVPDP_bev_div_{}.npy'.format(div)
            # np.save(filename, statebehaviors)

            time_list = statebehaviors[:, 0]
            U_list = statebehaviors[:, 1]
            rho_list = statebehaviors[:, 2]
            I_list = statebehaviors[:, 3]
            D_list = statebehaviors[:, 4]
            V_list = statebehaviors[:, 5]

            PathOutDir = 'RecursiveMth_Output'
            if not os.path.exists(PathOutDir):
                os.mkdir(PathOutDir)

            # Boundary Value Problem, change 'div' from 10 to 16
            filename = 'DiffGame_BVPDynPrgRecur_div{}'.format(div)
            filepath = os.path.join(PathOutDir, filename)
            np.savez(filepath, div=div, h=h, T=T, N=N, time_list = time_list,
                     I_list = I_list, D_list=D_list, U_list=U_list, rho_list=rho_list)

            # timestr = t_begin.strftime('%Y%m%d%H%M%S')
            filename = 'outputstr_DiffGame_BVPDynPrgRecur_div{}.txt'.format(div)
            filepath = os.path.join(PathOutDir, filename)
            f = open(filepath, mode='w+')
            f.write(outputstr)
            f.close()

            div = div + 1

