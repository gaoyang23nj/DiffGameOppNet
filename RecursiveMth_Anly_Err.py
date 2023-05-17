# coding = utf-8
# The Python program is to calculate
# 'the upper bound' and 'the lower bound' of the analytical results
# according to the error analysis theory of the differential equations.

# This Program reads
# ./RecursiveMth_Output/DiffGame_BVPDynPrgRecur_div{}.npz (the result npz file),
# and outputs:
# RecursiveMth_AnalyRes_div{}.csv (the result csv file),
# which also includes the mean estimation and the error estimation;

import sympy
import numpy as np
import os

# Tips:
# assume that T/h = K
# original list len(I_list) = K from t=0 to t=K-1
# output list len(I_test) = K+1 from t=0 to t=K,
# D_list and D_test are similar with them.
# Because function V() cannot provide I_K, D_K at t=T
# function get_NashEq_list() can deduce I_K, D_K at t=T via the behaviors.
def get_NashEq_list(filename, lam, N, gamma, alpha, beta):
    def get_derivate_ID(tmp_U, tmp_rho, I, D):
        derivate_I = lam * (N - I) - tmp_rho * I / N
        derivate_D = tmp_rho * I / N - lam * D - 1 / N * D * tmp_U
        return derivate_I, derivate_D

    cal_res = np.load(filename)
    time_list = cal_res['time_list']
    U_list = cal_res['U_list']
    rho_list = cal_res['rho_list']
    I_list = cal_res['I_list']
    D_list = cal_res['D_list']
    I_test = [0]
    D_test = [0]
    # differential -> difference 微分转差分
    # 按照U rho h 从正向计算一下 I_test D_test
    assert len(U_list) == len(rho_list)
    tmp_cost = 0.
    for i in range(len(U_list)):
        U = U_list[i]
        rho = rho_list[i]
        I = I_test[-1]
        D = D_test[-1]
        derivate_I, derivate_D = get_derivate_ID(U, rho, I, D)
        Ik1 = I + h * derivate_I
        Dk1 = D + h * derivate_D
        tmp_cost = tmp_cost + (gamma * (D+Dk1)/2 + alpha * U - beta * rho)*h
        I_test.append(Ik1)
        D_test.append(Dk1)
    return time_list, I_test, D_test, U_list, rho_list

lam = 0.004
N = 100
gamma = 1
alpha = 1
beta = 0.1

# use the symbol, then can get the values with varying variable (e.g., rho, U, I, D).
rho = sympy.Symbol('rho')
U = sympy.Symbol('U')
I = sympy.Symbol('I')
D = sympy.Symbol('D')

# div = 10
div = 16
T = 500
h = T / div

################################# tigher bound for I
print('*'*200)

PathOutDir = '../RecursiveMth_Output'
filename = 'DiffGame_BVPDynPrgRecur_div{}.npz'.format(div)
filepath = os.path.join(PathOutDir, filename)

time_list, I_list_ex, D_list_ex, U_list, rho_list = get_NashEq_list(filepath, lam, N, gamma, alpha, beta)
time_list_ex = np.zeros(div + 1, dtype='float')
time_list_ex[:-1] = time_list
time_list_ex[-1] = time_list[-1] + h
print('time_list_ex', time_list_ex)
print('I_list_ex', I_list_ex)
print('D_list_ex', D_list_ex)
print('U_list', U_list)
print('rho_list', rho_list)

# Section 1: calculate the upper bound of I
term_I = 1 - h * (lam + rho/N)
M_I = (lam + rho/N)*(lam*N - (lam+rho/N) *I)

err_I_0 = 0
list_sup_err_I = [err_I_0]
for i in range(div):
    value_term_I = term_I.subs([(rho, rho_list[i])])
    value_M_I = M_I.subs([(rho, rho_list[i]), (I, I_list_ex[0])])
    err_tmp = value_term_I * list_sup_err_I[-1] + (h*h)/2 * value_M_I
    list_sup_err_I.append(err_tmp)
print('list_sup_err_I, len:{}'.format(len(list_sup_err_I)))
print(list_sup_err_I)

# upper bound list of I
I_upper_bound_list = []
for i in range(div+1):
    tmp_upper = I_list_ex[i] + list_sup_err_I[i]
    if tmp_upper > N:
        tmp_upper = N
    I_upper_bound_list.append(tmp_upper)

# lower bound list of I
I_lower_bound_list = []
for i in range(div+1):
    tmp_lower = I_list_ex[i] - list_sup_err_I[i]
    if tmp_lower < 0:
        tmp_lower = 0
    I_lower_bound_list.append(tmp_lower)

print(I_upper_bound_list)
print(I_list_ex)
print(I_lower_bound_list)


# Section 2: calculate the upper bound of D
term_D = 1 - h * (lam + U/N)
# note that here D should substuite
M_D = rho/N * (lam*N - lam*I - rho/N*I) \
      - (lam + U/N) * (rho/N * I - lam * D - U/N * D)

err_D_0 = 0
list_sup_err_D = [err_D_0]
for i in range(div):
    value_term_D = term_D.subs([(U, U_list[i])])
    value_M_D = M_D.subs([(rho, rho_list[i]), (U, U_list[i]),
                          (I, I_list_ex[0]), (D, D_list_ex[i+1])])
    err_tmp = value_term_D * list_sup_err_D[-1] + (h*h)/2 * value_M_D
    list_sup_err_D.append(err_tmp)
print('list_sup_err_D, len:{}'.format(len(list_sup_err_D)))
print(list_sup_err_D)


# upper bound of I
D_upper_bound_list = []
for i in range(div+1):
    tmp_upper = D_list_ex[i] + list_sup_err_D[i]
    if tmp_upper > N:
        tmp_upper = N
    D_upper_bound_list.append(tmp_upper)

# lower bound of I
D_lower_bound_list = []
for i in range(div+1):
    tmp_lower = D_list_ex[i] - list_sup_err_D[i]
    if tmp_lower < 0:
        tmp_lower = 0
    D_lower_bound_list.append(tmp_lower)

print(D_upper_bound_list)
print(D_list_ex)
print(D_lower_bound_list)


# Section 3: convert to string
def ConvertoString(ll):
    output_str = ''
    for i in range(len(ll)-1):
        output_str = output_str + str(ll[i]) + ','
    output_str = output_str + str(ll[len(ll)-1]) + '\n'
    return output_str


filename = 'RecursiveMth_AnalyRes_div{}.csv'.format(div)
f = open(filename, 'w+')
f.write(ConvertoString(time_list_ex))
f.write(ConvertoString(I_list_ex))
f.write(ConvertoString(D_list_ex))

f.write(ConvertoString(I_upper_bound_list))
f.write(ConvertoString(I_lower_bound_list))

f.write(ConvertoString(D_upper_bound_list))
f.write(ConvertoString(D_lower_bound_list))
f.write(ConvertoString(U_list))
f.write(ConvertoString(rho_list))
f.close()