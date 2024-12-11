# 1.given U(t), div=
# 2.calculate best rho(t)
# 3.calculate the analytical result
import datetime
import os

import numpy as np
isDraw = False

# # the number of nodes
# N = 100
# # the frequency of node concacts
# lam = 0.004
#
# # --20241201 infocom'05_dataset # id:1 num_interval:644 avg_value:383.01863354037266 estimated_Lambda:0.0026108390360976876
# N = 40
# lam = 0.0024484716060355475
#
# # --20241203 RWP_dataset # id:0 num_interval:2754 avg_value:313.4738562091503 estimated_Lambda:0.003190058692909938
# N = 99
# lam = 0.003174182967121075
#
# # --20241204 Helsinki_dataset id:20 num_interval:112 avg_value:629.7410714285714 estimated_Lambda:0.001587954232890502
N = 99
lam = 0.001702483720762598

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


def get_derivate_ID(tmp_U, tmp_rho, I, D):
    derivate_I = lam*(N-I)-tmp_rho*I/N
    derivate_D = tmp_rho*I/N-lam*D-1/N*D*tmp_U
    return derivate_I, derivate_D

# main alg., to obtain the best reply, i.e., rho(t) for the given U(t)
def BestReply(t, I, D, U):
    # print('in function V(t,I,D) begin t：{}, I:{}, D:{}'.format(t, I, D))
    if t>=T:
        return 0, []
    # find min max V(t, I, D) with the Comparing these four situations.
    # [four situations: (U_min, rho_min),(U_min, rho_max),(U_max, rho_min),(U_max,rho_max)]
    # tmp value is stored in list_value (Python LIST []).
    list_value = []

    for tmp_rho in [rho_min, rho_max]:
        tmp_U = U[int(t/h)]

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
        term1, t_next_behaviors = BestReply(t+h, I_new, D_new, U)

        # get the Cost at this time (t)
        # Choose the Equilibrium Solution with Comparision Method
        value = term1 + term2
        list_value.append((value, tmp_U, tmp_rho, t_next_behaviors))
    # print(list_value[0][0], list_value[1][0], list_value[2][0], list_value[3][0])

    # rho(t) wishes to maximize J
    if list_value[0][0] > list_value[1][0]:
        t_behaviors_new = list_value[0][3].copy()
        # Insert the current behaviors (U and rho), the current state (I and D), the value to the List.
        t_behaviors_new.insert(0, (t, list_value[0][1], list_value[0][2], I, D, list_value[0][0]))
        return list_value[0][0], t_behaviors_new
    else:
        t_behaviors_new = list_value[1][3].copy()
        t_behaviors_new.insert(0, (t, list_value[1][1], list_value[1][2], I, D, list_value[1][0]))
        return list_value[1][0], t_behaviors_new

def ObtainPrintBestrho(U_list):
    # calculate best rho(t)
    # based on the control (U and rho) and the state (I and D),
    # we can get the derivate of I and D
    # recursive
    print('T:{} div:{}'.format(T, div))
    print('Umax:{} rhomax:{}'.format(U_max, rho_max))
    outputstr = 'T:{} div:{}\nU_max:{} rho_max:{}\n'.format(T, div, U_max, rho_max)
    t_begin = datetime.datetime.now()
    print('begin at: {}'.format(t_begin))
    # conduct the computation of the NE strategy.
    value, t_behaviors = BestReply(0, I0, D0, U_list)
    t_end = datetime.datetime.now()
    print('end at: {}'.format(t_end))
    delta_t = t_end - t_begin
    print(delta_t)
    # record computation time
    # ct_file = open(computation_time_file,'a+')
    # ct_file.write('{},{},{}\n'.format(div, delta_t.total_seconds(), delta_t))
    # ct_file.close()
    outputstr = outputstr + 'begin at: {}\n'.format(t_begin) + 'end at: {}\n'.format(t_end) + '{}\n'.format(delta_t)
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
    PathOutDir = 'RecursiveMth_Output_CompareU'
    if not os.path.exists(PathOutDir):
        os.mkdir(PathOutDir)
    # Boundary Value Problem, change 'div' from 10 to 16
    filename = 'DiffGame_BVPDynPrgRecur_div{}'.format(div)
    filepath = os.path.join(PathOutDir, filename)
    np.savez(filepath, div=div, h=h, T=T, N=N, time_list=time_list,
             I_list=I_list, D_list=D_list, U_list=U_list, rho_list=rho_list)
    # timestr = t_begin.strftime('%Y%m%d%H%M%S')
    filename = 'outputstr_DiffGame_BVPDynPrgRecur_div{}.txt'.format(div)
    filepath = os.path.join(PathOutDir, filename)
    f = open(filepath, mode='w+')
    f.write(outputstr)
    f.close()

    if isDraw:
        import matplotlib.pyplot as plt
        fig_list = np.zeros((2 * div, 3))
        for i in range(div):
            fig_list[i * 2, 0] = time_list[i]
            fig_list[i * 2 + 1, 0] = time_list[i] + 0.99 * h
            fig_list[i * 2, 1] = U_list[i]
            fig_list[i * 2 + 1, 1] = U_list[i]
            fig_list[i * 2, 2] = rho_list[i]
            fig_list[i * 2 + 1, 2] = rho_list[i]
        plt.figure(1)
        plt.plot(fig_list[:, 0], fig_list[:, 1], '-o', 'b')
        plt.plot(fig_list[:, 0], fig_list[:, 2], '-d', 'r')
        plt.show()

    return value, rho_list

candidate_choices = []
for i in [U_min, U_max]:
    for j in [rho_min, rho_max]:
        candidate_choices.append((i, j))

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

    print('Internal error! V()-- t:{}, I:{}, D:{}'.format(t,I,D))
    # t_behaviors_new = list_value[3][3].copy()
    # t_behaviors_new.insert(0, (t, list_value[3][1], list_value[3][2], I, D, list_value[3][0]))
    # return list_value[3][0], t_behaviors_new
    # combined strategy
    # # probality of rho_off
    # p_rho = (list_value[3][0] - list_value[1][0])/(list_value[0][0] - list_value[1][0] - list_value[2][0] + list_value[3][0])
    # # list_value[0][0]*p_rho + list_value[1][0]*(1-p_rho) == list_value[2][0]*p_rho + list_value[3][0]*(1-p_rho)
    # # probality of U_off
    # p_U = (list_value[3][0] - list_value[2][0]) / (list_value[0][0] - list_value[2][0] - list_value[1][0] + list_value[3][0])
    # # list_value[0][0]*p_U + list_value[2][0]*(1-p_U) == list_value[1][0]*p_U + list_value[3][0]*(1-p_U)
    # tmp_U = p_U * U_min + (1 - p_U) * U_max
    # tmp_rho = p_rho * rho_min + (1 - p_rho) * rho_max
    # # # based on the current state and the employed controls,
    # # # we can obtain the new I and the new D
    # derivate_I, derivate_D = get_derivate_ID(tmp_U, tmp_rho, I, D)
    # I_new = I + h * derivate_I
    # D_new = D + h * derivate_D
    # print('p_Uoff:{} p_rhooff:{}'.format(p_U, p_rho))
    # # get the Running Cost in the time duration [t, t+h]
    # term2 = h * (c * (D + D_new) / 2 + alpha * tmp_U - beta * tmp_rho)
    # tmp_value = list_value[0][0]*p_U + list_value[2][0]*(1-p_U)
    # # 重构t~T
    # term1, t_next_behaviors = V(t + h, I_new, D_new)
    # t_behaviors_new = t_next_behaviors.copy()
    # value = term1 + term2
    # t_behaviors_new.insert(0, (t, tmp_U, tmp_rho, I, D, value))
    # loop situation game

    # # random strategy
    # p_rho = (list_value[3][0] - list_value[1][0])/(list_value[0][0] - list_value[1][0] - list_value[2][0] + list_value[3][0])
    # # list_value[0][0]*p_rho + list_value[1][0]*(1-p_rho) == list_value[2][0]*p_rho + list_value[3][0]*(1-p_rho)
    # # probality of U_off
    # p_U = (list_value[3][0] - list_value[2][0]) / (list_value[0][0] - list_value[2][0] - list_value[1][0] + list_value[3][0])
    # # list_value[0][0]*p_U + list_value[2][0]*(1-p_U) == list_value[1][0]*p_U + list_value[3][0]*(1-p_U)
    # _x = np.random.random(2)
    # idx = 2*int(_x[0] >= p_U)+int(_x[1] >= p_rho)
    # t_behaviors_new = list_value[idx][3].copy()
    # t_behaviors_new.insert(0, (t, list_value[idx][1], list_value[idx][2], I, D, list_value[idx][0]))

    #
    # # random strategy
    # p_rho = (list_value[3][0] - list_value[1][0])/(list_value[0][0] - list_value[1][0] - list_value[2][0] + list_value[3][0])
    # # list_value[0][0]*p_rho + list_value[1][0]*(1-p_rho) == list_value[2][0]*p_rho + list_value[3][0]*(1-p_rho)
    # # probality of U_off
    # p_U = (list_value[3][0] - list_value[2][0]) / (list_value[0][0] - list_value[2][0] - list_value[1][0] + list_value[3][0])
    # # list_value[0][0]*p_U + list_value[2][0]*(1-p_U) == list_value[1][0]*p_U + list_value[3][0]*(1-p_U)
    # idx = 2*int(p_U < 0.5)+int(p_rho < 0.5)
    # t_behaviors_new = list_value[idx][3].copy()
    # t_behaviors_new.insert(0, (t, list_value[idx][1], list_value[idx][2], I, D, list_value[idx][0]))
    #
    # return list_value[idx][0], t_behaviors_new


# the whole simulation duration
T_list = [100]
# T_list = [20000, 50000, 100000]
# T_list = [300, 400, 500, 600, 700, 800, 900, 1000]
# T_list = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
# T_list = [300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100]
J_result = np.zeros((5, len(T_list)))
div = 10
for T_idx in range(len(T_list)):
    T = T_list[T_idx]
    # T = 500
    x = np.linspace(0, T, div + 1)
    h = T/div

    # (1) given U(t), method_1, U_start = 100
    value, t_behaviors = V(0, I0, D0)
    # Arrange the results into
    U_list = np.zeros((len(t_behaviors)), dtype='float')
    for i in range(len(t_behaviors)):
        # # time t
        # statebehaviors[i][0] = t_behaviors[i][0]
        # U(t), the control
        U_list[i] = t_behaviors[i][1]
        # # rho(t), the control
        # statebehaviors[i][2] = t_behaviors[i][2]
        # # I(t), the state
        # statebehaviors[i][3] = t_behaviors[i][3]
        # # D(t), the state
        # statebehaviors[i][4] = t_behaviors[i][4]
        # V(t), value
        # statebehaviors[i][5] = t_behaviors[i][5]
    U_method1 = U_list
    # U_method1 = np.ones(div+1)*U_max
    # #U_start = 100
    # #U(0) U(50)=0, U(100)=U_max
    # U_method1[:2] = U_min
    print('*'*20+':NE')
    print(U_method1.tolist())
    value, rho_list = ObtainPrintBestrho(U_method1)
    print(rho_list.tolist())
    J_result[0, T_idx] = value

    # (2) given U(t), method_1, Umin
    U_method2 = np.ones(div+1)*U_min
    print('*'*20+'Umin')
    print(U_method2.tolist())
    value, rho_list = ObtainPrintBestrho(U_method2)
    print(rho_list.tolist())
    J_result[1, T_idx] = value

    # (3) given U(t), method_1, Umax
    U_method3 = np.ones(div+1)*U_max
    print('*'*20+'Umax')
    print(U_method3.tolist())
    value, rho_list = ObtainPrintBestrho(U_method3)
    print(rho_list.tolist())
    J_result[2, T_idx] = value

    # (4) given U(t), method_1, half Umax
    U_method4 = np.ones(div+1)*U_max*0.5
    print('*'*20+'half Umax')
    print(U_method4.tolist())
    value, rho_list = ObtainPrintBestrho(U_method4)
    print(rho_list.tolist())
    J_result[3, T_idx] = value

    # (5) given U(t), method_1, 'on-off-on-off-...'
    U_method5 = np.ones(div+1)*U_max
    for i in range(len(U_method5)):
        if i%2==1:
            U_method5[i]=U_min
    print('*'*20+'on-off-on')
    print(U_method5.tolist())
    value, rho_list = ObtainPrintBestrho(U_method5)
    print(rho_list.tolist())
    J_result[4, T_idx] = value

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(T_list, J_result[0,:], 'r-o')
plt.plot(T_list, J_result[1,:], 'b-^')
plt.plot(T_list, J_result[2,:], 'k-s')
plt.plot(T_list, J_result[3,:], 'g-d')
plt.plot(T_list, J_result[4,:], 'y-v')
plt.show()

# import pandas as pd
# varyingT_list = np.vstack((T_list, J_result))
# df = pd.DataFrame(varyingT_list)
# df.to_csv('CompareU_bestRho_varyingT.csv',index= False, header= False)


# print('*'*20+'customized')
# U_method1 = [1.0]*10
# U_method1[0] = 0
# print(U_method1)
# value, rho_list = ObtainPrintBestrho(U_method1)
# print(rho_list.tolist())
# print(value)