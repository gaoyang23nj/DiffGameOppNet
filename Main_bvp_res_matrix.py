# 遍历所有结果
# 3D图像呈现
# 唯一的变化是U_begin
import scipy.integrate
import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt



T = 500
lam = 0.004
N = 100

c = 1
alpha = 1
beta = 0.1

U_max = 1
U_min = 0
rho_max = 0.5
rho_min = 0



def get_derivate(I, D,tmp_U, tmp_rho):
    derivate_I = lam*(N-I)-tmp_rho*I/N
    derivate_D = tmp_rho*I/N-lam*D-D*tmp_U/N
    return derivate_I, derivate_D

def cal_ana_res(tmp_U, tmp_rho, h, isDraw):
    # print(tmp_U)
    # print(tmp_rho)
    # x = np.linspace(0, T, div)
    # y = np.zeros((2, x.size))
    I = [0]
    D = [0]
    # 演化起来
    assert tmp_U.size==tmp_rho.size
    t = 0
    i = 0
    while t<T:
        # 计算瞬时导数
        derivate_I, derivate_D = get_derivate(I[-1], D[-1], tmp_U[i], tmp_rho[i])
        Inew = I[-1]+derivate_I*h
        Dnew = D[-1]+derivate_D*h
        I.append(Inew)
        D.append(Dnew)
        t = t+h
        i = i+1
    # r = N - y[0,:] - y[1,:]
    # i = y[0,:]
    # d = y[1,:]
    assert len(I)-1 == tmp_U.size
    # res = get_sim_payoff(N, r, y[0,:], y[1,:], U, rho)
    res = (c*sum(D[:-1])+alpha*sum(tmp_U)-beta*sum(tmp_rho))*h
    print('res:{}'.format(res))
    print('sum_U:{} sum_rho:{}'.format(np.sum(tmp_U), np.sum(tmp_rho)))
    return res, np.sum(tmp_U), np.sum(tmp_rho)

t_on_U = t_on_rho = 0
t_off_U = t_off_rho = T

# 计算
div = 500
time = np.linspace(0, T, div+1)
h = T/div

stepsize = 10
# grids = 50
# size1 = np.linspace(0, T, grids)
# size2 = np.linspace(0, T, grids)
# size1 = len(range(0, T, stepsize))
# size2 = len(range(0, T, stepsize))
U_grid = np.arange(t_on_U, t_off_U, stepsize)
rho_grid = np.arange(t_on_rho, t_off_U, stepsize)
matrix_res = np.ones((U_grid.size, rho_grid.size))*(-1)

i = 0
for begin_U in U_grid:
    # from begin_U to t_off_U
    tmp_U = np.zeros(div)
    for k in range(tmp_U.size):
        if time[k] >= begin_U and time[k] <= t_off_U:
            tmp_U[k] = 1
    # tmp_U[begin_U:t_off_U]=1
    j = 0
    for begin_rho in rho_grid:
        # tmp_rho = np.zeros(T)
        # tmp_rho[begin_rho:t_off_rho] = 0.5
        tmp_rho = np.zeros(div)
        for k in range(tmp_rho.size):
            if time[k] >= begin_rho and time[k] <= t_off_rho:
                tmp_rho[k] = 0.5
        print(begin_U, begin_rho)
        res, sum1, sum2 = cal_ana_res(tmp_U.copy(), tmp_rho.copy(), h, False)
        matrix_res[i,j] = res
        j = j + 1
    i = i+1

print(matrix_res)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# 画出矩阵
fig = plt.figure()
ax = fig.gca(projection='3d')
# U_grid: [0, 10, 20, ... , 500]
# rho_grid: [0, 10, 20, ... , 500]
# rho:[[0, 10, 20, ... , 500],
#      [0, 10, 20, ... , 500],
#      ...,
#      [0, 10, 20, ... , 500]]
# U:[[0, 0, 0..., 0],
#    [10, 10, 10, ... 10],]
#    ...,
#    [500, 500, 500, ..., 500]]
rho, U = np.meshgrid(rho_grid, U_grid)
surf = ax.plot_surface(U, rho, matrix_res, cmap=cm.coolwarm, linewidth=0, antialiased=False)
min_res = np.min(np.min(matrix_res))

# K = 10
begin_U_K10 = 100
begin_rho_K10 = 50
tmp_rho = np.zeros(div)
tmp_U = np.zeros(div)
for k in range(tmp_U.size):
    if time[k] >= begin_U_K10 and time[k] <= t_off_U:
        tmp_U[k] = 1
for k in range(tmp_rho.size):
    if time[k] >= begin_rho_K10 and time[k] <= t_off_rho:
        tmp_rho[k] = 0.5
print(begin_U_K10, begin_rho_K10)
res_K10, sum1, sum2 = cal_ana_res(tmp_U, tmp_rho, h, False)

ax.scatter(begin_U_K10, begin_rho_K10, res_K10+400, c='r', marker='*')
ax.scatter(begin_U_K10, begin_rho_K10, min_res-400, c='b', marker='*')
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('U_begin')
ax.set_ylabel('rho_begin')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

begin_U_K16 = 500./16.*2.
begin_rho_K16 = 500/16
tmp_rho = np.zeros(div)
tmp_U = np.zeros(div)
for k in range(tmp_U.size):
    if time[k] >= begin_U_K16 and time[k] <= t_off_U:
        tmp_U[k] = 1
for k in range(tmp_rho.size):
    if time[k] >= begin_rho_K16 and time[k] <= t_off_rho:
        tmp_rho[k] = 0.5
print(begin_U_K16, begin_rho_K16)
res_K16, sum1, sum2 = cal_ana_res(tmp_U, tmp_rho, h, False)

# store the results into the file
import pandas as pd
import numpy as np
df = pd.DataFrame(U)
df.to_csv('3Dfig_matrix_U.csv',index= False, header= False)
df = pd.DataFrame(rho)
df.to_csv('3Dfig_matrix_rho.csv',index= False, header= False)
df = pd.DataFrame(matrix_res)
df.to_csv('3Dfig_matrix_res.csv',index= False, header= False)
NE_solution_K10 = [begin_U_K10, begin_rho_K10, res_K10]
NE_solution_K16 = [begin_U_K16, begin_rho_K16, res_K16]
df = pd.DataFrame(NE_solution_K10)
df.to_csv('3Dfig_matrix_NEsolutionK10.csv',index= False, header= False)
df = pd.DataFrame(NE_solution_K16)
df.to_csv('3Dfig_matrix_NEsolutionK16.csv',index= False, header= False)




