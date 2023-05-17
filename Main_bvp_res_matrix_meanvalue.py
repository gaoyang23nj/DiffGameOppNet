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

gamma = 1

para_time_granularity = 0.1

event_fromsrc = 'rcv_from_src'
event_selfish = 'to_be_selfish'
event_detect = 'detect'

state_without_message = 0
state_with_message = 1
state_selfish = 2

class Sim(object):
    def __init__(self, num_nodes, total_time, lam, h, U_star, rho_star):
        self.lam = lam
        # 检测策略
        self.U_star = U_star
        # 自私策略
        self.rho_star = rho_star
        self.T = total_time
        self.N = num_nodes
        self.h = h
        # 当前运行时间
        self.running_time = 0
        # 总体上下一次相遇事件的时刻
        self.list_nextContact = []
        # 假定的selfish源头
        self.sel_nextContact = np.ones(self.N, dtype='float') * -1
        # 假定message源头
        self.src_nextContact = np.ones(self.N, dtype='float') * -1

        # 假定检查时间
        # self.detect_nextContact = 0.

        # 每个node的状态 -- 矩阵
        # 0 表示 without_message R; 1 表示with_message I; 2 表示selfish状态 D
        self.stateNode = np.zeros(self.N, dtype='int')

        # 初始化next_contact_time矩阵
        self.__init_contact_time()

        # 中间结果
        self.res_record = []
        # 保存结果 0.1->10 1个单位事件对应多少个granularity
        self.per = int(1/para_time_granularity)
        # 方便画图, 为时间粒度加上编号，并给出准确的时刻
        self.time_index = np.arange(0, self.T*self.per)
        self.true_time = self.time_index*para_time_granularity
        # 3种状态的节点个数统计
        self.res_nr_nodes_no_message = np.ones(self.T*self.per, dtype='int') * -1
        self.res_nr_nodes_with_message = np.ones(self.T*self.per, dtype='int') * -1
        self.res_nr_nodes_selfish = np.ones(self.T*self.per, dtype='int') * -1
        # 为了便于仿真统计 初始时刻的状态 需要记录一下
        (t, nr_r, nr_i, nr_d) = self.update_nr_nodes_record_with_time()
        self.res_nr_nodes_no_message[0] = nr_r
        self.res_nr_nodes_with_message[0] = nr_i
        self.res_nr_nodes_selfish[0] = nr_d
        while True:
            # next_time > total_sim
            if self.list_nextContact[0][0] >= self.T:
                break
            # 相遇投递 新投递时间更新 selfish变异
            self.run()

        # 整理结果 从self.res_record刷到self.res_nr_nodes_no_message等3个矩阵中
        i = 1
        # time = 0 + para_time_granularity
        for i in range(1, self.true_time.size):
            is_found = False
            for j in range(len(self.res_record)):
                (t, nr_r, nr_i, nr_d) = self.res_record[j]
                if t > self.true_time[i]:
                    break
                # if t <= time:
                # 对于record里面比当前时刻小的，可以不停的复制，直到停止
                else:
                    is_found = True
                    self.res_nr_nodes_no_message[i] = nr_r
                    self.res_nr_nodes_with_message[i] = nr_i
                    self.res_nr_nodes_selfish[i] = nr_d
            # 一般地，对于最后一个时刻记录而言，必须使用之前的状态
            if not is_found:
                self.res_nr_nodes_no_message[i] = self.res_nr_nodes_no_message[i - 1]
                self.res_nr_nodes_with_message[i] = self.res_nr_nodes_with_message[i - 1]
                self.res_nr_nodes_selfish[i] = self.res_nr_nodes_selfish[i - 1]

    @staticmethod
    def get_next_wd(pl):
        res = np.random.exponential(1 / pl)
        return res

    def __init_contact_time(self):
        # 节点收到message
        # 为每个节点计算下次相遇时刻
        for i in range(self.N):
            # 对于i节点 与src相遇的时刻为
            self.src_nextContact[i] = self.get_next_wd(self.lam)
            # 记录到事件列表
            self.list_nextContact.append((self.src_nextContact[i], event_fromsrc, i))

        # 检测活动本身是按次计算的，1次检测有意义，0.5次检测没意义
        tmp_list_U = []
        assert self.h * len(self.U_star) == self.T
        # 转化为 (time, U) ... list的形式
        for i in range(len(self.U_star)):
            # 在 时刻i*h到时刻(i+1)*h 之间
            time = i*self.h
            tmp_list_U.append((time, self.U_star[i]))
        timeindex = 0.
        tmp_list_U_final = []
        while timeindex < self.T:
            newvalue = -1
            if timeindex >= tmp_list_U[-1][0]:
                newvalue = tmp_list_U[-1][1]
            else:
                for j in range(len(tmp_list_U)-1):
                    if timeindex >= tmp_list_U[j][0] and timeindex < tmp_list_U[j+1][0]:
                        newvalue = tmp_list_U[j][1]
                        break
            assert newvalue > -1
            tmp_list_U_final.append((timeindex, newvalue))
            timeindex = timeindex + para_time_granularity
        tmp = 0.
        for (timeindex, value) in tmp_list_U_final:
            tmp = tmp + value*para_time_granularity
            if tmp >= 1.:
                self.list_nextContact.append((timeindex, event_detect))
                tmp = tmp - 1

        tmp_list_rho = []
        # 转化为 (time, U) ... list的形式
        assert self.h*len(self.rho_star) == self.T
        for i in range(len(self.rho_star)):
            # 在 时刻i*h到时刻(i+1)*h 之间
            time = i * self.h
            tmp_list_rho.append((time, self.rho_star[i]))
        timeindex = 0.
        tmp_list_rho_final = []
        while timeindex < self.T:
            newvalue = -1
            if timeindex >= tmp_list_rho[-1][0]:
                newvalue = tmp_list_rho[-1][1]
            else:
                for j in range(len(tmp_list_rho) - 1):
                    if timeindex >= tmp_list_rho[j][0] and timeindex < tmp_list_rho[j + 1][0]:
                        newvalue = tmp_list_rho[j][1]
                        break
            assert newvalue > -1
            tmp_list_rho_final.append((timeindex, newvalue))
            timeindex = timeindex + para_time_granularity
        tmp = 0.
        for (timeindex, value) in tmp_list_rho_final:
            tmp = tmp + value * para_time_granularity
            if tmp >= 1.:
                self.list_nextContact.append((timeindex, event_selfish))
                tmp = tmp - 1
        # 按时间排序
        self.list_nextContact.sort()

    def update_nr_nodes_record_with_time(self):
        nr_r, target_list = self.get_sel_state_node(state=state_without_message)
        # self.res_nr_nodes_no_message[time_idx] = nr_h
        nr_i, target_list = self.get_sel_state_node(state=state_with_message)
        # self.res_nr_nodes_with_message[time_idx] = nr_i
        nr_d, target_list = self.get_sel_state_node(state=state_selfish)
        # self.res_nr_nodes_selfish[time_idx] = nr_m
        return self.running_time, nr_r, nr_i, nr_d

    def get_sel_state_node(self, state):
        tmp_i = np.sum(self.stateNode == state)
        # 获取对应的 index_list
        index_list = np.argwhere(self.stateNode == state)
        index_list = np.squeeze(index_list, axis=1)
        index_list = index_list.tolist()
        # 选择一个节点(选第一个) // 现在暂时不考虑 src不变异的假设
        np.random.shuffle(index_list)
        return tmp_i, index_list

    def run(self):
        if self.list_nextContact[0][1] == event_detect:
            # print(self.list_nextContact[0])
            # print(self.running_time, self.list_nextContact)
            (t, eve) = self.list_nextContact[0]
            assert self.running_time <= t
            # 更新新的时间 和 pop 事件
            self.running_time = t
            self.list_nextContact.pop(0)
            target_detect = np.random.randint(0, self.N)
            if self.stateNode[target_detect] == state_selfish:
                self.stateNode[target_detect] = state_without_message
            self.res_record.append(self.update_nr_nodes_record_with_time())
        elif self.list_nextContact[0][1] == event_fromsrc:
            # print(self.list_nextContact[0])
            # print(self.running_time, self.list_nextContact)
            (t, eve, i) = self.list_nextContact[0]
            assert self.running_time <= t
            # 更新新的时间 和 pop 事件
            self.running_time = t
            self.list_nextContact.pop(0)

            if self.stateNode[i] != state_with_message:
                self.stateNode[i] = state_with_message
            self.update_to_from_src(i)
            self.res_record.append(self.update_nr_nodes_record_with_time())
        elif self.list_nextContact[0][1] == event_selfish:
            # print(self.list_nextContact[0])
            # print(self.running_time, self.list_nextContact)
            (t, eve) = self.list_nextContact[0]
            assert self.running_time <= t
            # 更新新的时间 和 pop 事件
            self.running_time = t
            self.list_nextContact.pop(0)
            target_detect = np.random.randint(0, self.N)
            if self.stateNode[target_detect] == state_with_message:
                self.stateNode[target_detect] = state_selfish
            self.res_record.append(self.update_nr_nodes_record_with_time())
            # self.res_record.append(self.update_nr_nodes_record_with_time())
        else:
            print('Internal Err! -- unkown event time:{} eve_list:{}'.format(self.running_time, self.list_nextContact))

    def update_to_from_src(self, i):
        tmp_next_time = self.get_next_wd(self.lam) + self.running_time
        self.src_nextContact[i] = tmp_next_time
        loc = 0
        for loc in range(len(self.list_nextContact)):
            if self.list_nextContact[loc][0] >= tmp_next_time:
                break
        self.list_nextContact.insert(loc, (tmp_next_time, event_fromsrc, i))

    # 提供时间维度 以及各时刻各状态节点的个数
    def get_sim_res(self):
        return self.time_index, self.true_time, self.res_nr_nodes_no_message, \
               self.res_nr_nodes_with_message, self.res_nr_nodes_selfish


# 计算payoff
def get_sim_payoff(N, T, r, i, d, U, rho, h):
    K0 = np.sum(d) * gamma * para_time_granularity
    K1 = np.sum(U) * alpha * h
    K2 = np.sum(rho) * beta * h
    res = K0 + K1 - K2
    return res

def cal_mean_res(U_list, rho_list, h):
    run_times = 20
    multi_times_payoff = np.zeros(run_times)
    for k in range(run_times):
        print('running time:{}'.format(k))
        the = Sim(N, T, lam, h, U_list, rho_list)
        x, x_true, r, i, d = the.get_sim_res()
        payoff = get_sim_payoff(the.N, T, r, i, d, U_list, rho_list, h)
        # print('payoff:{}'.format(payoff))
        multi_times_payoff[k] = payoff
        # print('cost_from_sel:{} cost_from_detect:{}'.format(cost1, cost2))
    mean_payoff = np.sum(multi_times_payoff, axis=0) / run_times
    print('mean payoff:{}'.format(mean_payoff))
    return mean_payoff

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
        res = cal_mean_res(tmp_U.copy(), tmp_rho.copy(), h)
        matrix_res[i,j] = res
        j = j + 1
    i = i+1

print(matrix_res)

# K=10
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
res_K10 = cal_mean_res(tmp_U.copy(), tmp_rho.copy(), h)

# K=16
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
res_K16 = cal_mean_res(tmp_U.copy(), tmp_rho.copy(), h)

np.savez('MatrixJ_meanvalue_div.npz', matrix_res=matrix_res, opt_res = res)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# 画出矩阵
fig = plt.figure()
ax = fig.gca(projection='3d')
rho, U = np.meshgrid(rho_grid, U_grid)
surf = ax.plot_surface(U, rho, matrix_res, cmap=cm.coolwarm, linewidth=0, antialiased=False)
min_res = np.min(np.min(matrix_res))

ax.scatter(begin_U_K10, begin_U_K10, res_K10+400, c='r', marker='*')
ax.scatter(begin_U_K10, begin_U_K10, min_res-400, c='b', marker='*')
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('U_begin')
ax.set_ylabel('rho_begin')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# store the results into the file
import pandas as pd
import numpy as np
df = pd.DataFrame(U)
df.to_csv('3Dfig_Sim_matrix_U.csv',index= False, header= False)
df = pd.DataFrame(rho)
df.to_csv('3Dfig_Sim_matrix_rho.csv',index= False, header= False)
df = pd.DataFrame(matrix_res)
df.to_csv('3Dfig_Sim_matrix_res.csv',index= False, header= False)
NE_solution_K10 = [begin_U_K10, begin_rho_K10, res_K10]
NE_solution_K16 = [begin_U_K16, begin_rho_K16, res_K16]
df = pd.DataFrame(NE_solution_K10)
df.to_csv('3Dfig_Sim_matrix_NEsolutionK10.csv',index= False, header= False)
df = pd.DataFrame(NE_solution_K16)
df.to_csv('3Dfig_Sim_matrix_NEsolutionK16.csv',index= False, header= False)



