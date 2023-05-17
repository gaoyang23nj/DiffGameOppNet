# coding=utf-8
# This Python program is to conduct the simulation
# with the events and the message dissemination.

# This Program reads
# RecursiveMth_AnalyRes_div{}.csv
# (which includes the mean estimation and the error estimation)
# outputs:
# RecursiveMth_AllSimRes_div{}.csv (the all result file),
# which also includes the mean estimation, the error estimation and the simulation results;

import os
import numpy as np
import matplotlib.pyplot as plt

lam = 0.004
# para_lambda = lam
N = 100
gamma = 1
alpha = 1
beta = 0.1

div = 16
T = 500
h = T/div

run_times = 50
isDraw = True

# the time granularity of the whole simulation duration
para_time_granularity = 0.1

# 3 events in the whole simulation
event_fromsrc = 'rcv_from_src'
event_selfish = 'to_be_selfish'
event_detect = 'detect'

# 3 states of all the nodes
state_without_message = 0
state_with_message = 1
state_selfish = 2

class Sim(object):
    def __init__(self, num_nodes, total_time, lam, h, U_star, rho_star):
        # the given detection strategy, which is a list
        self.U_star = U_star
        # the given selfish strategy, which is a list
        self.rho_star = rho_star

        # parameters
        self.lam = lam
        self.T = total_time
        self.N = num_nodes
        self.h = h

        # the current time in the simulation
        self.running_time = 0
        # the Event List (according to the temporal sequence),
        # i.e., [ (the next event time, the corresponding event), (), ()]
        # in the whole simulation (based on the Poisson Process)
        self.list_nextEvent = []
        # the assumed source of being selfish,
        # i.e., the time of begin_selfish_event to all the N nodes
        self.sel_nextContact = np.ones(self.N, dtype='float') * -1
        # the assumed source node of the message,
        # i.e., the time of contact event to all the N nodes
        self.src_nextContact = np.ones(self.N, dtype='float') * -1

        # the state matrix, which records the states of all the nodes.
        # 0 --> without message, R state
        # 1 --> with message, I state
        # 2 --> being selfish, D state
        self.stateNode = np.zeros(self.N, dtype='int')

        # Initialize the Event List, i.e., self.list_nextEvent, next_contact_time矩阵
        self.__init_contact_time()

        # Note that this list is to store the middle result.
        self.res_record = []
        # convert one unit time into granularity
        self.per = int(1/para_time_granularity)
        # In order to plot, add the index and the true time
        self.time_index = np.arange(0, self.T * self.per)
        self.true_time = self.time_index * para_time_granularity
        # the statistics of the number of nodes in '3' states, at increasing time
        # these 3 lists record the number of nodes in 3 states, which is also varying with time increasing.
        self.res_nr_nodes_no_message = np.ones(self.T*self.per, dtype='int') * -1
        self.res_nr_nodes_with_message = np.ones(self.T*self.per, dtype='int') * -1
        self.res_nr_nodes_selfish = np.ones(self.T*self.per, dtype='int') * -1
        # In order to stat and plot,
        # At the initial time (i.e., when t=0), record the number of nodes in the 3 different states.
        (t, nr_r, nr_i, nr_d) = self.update_nr_nodes_record_with_time()
        self.res_nr_nodes_no_message[0] = nr_r
        self.res_nr_nodes_with_message[0] = nr_i
        self.res_nr_nodes_selfish[0] = nr_d
        while True:
            # if next_time > total_sim, the simulation is done
            if self.list_nextEvent[0][0] >= self.T:
                break
            # conduct the 3 events (contact, being selfish, detection); update the running time;
            self.run()

        # Convert the Results from self.res_record to 3 matrix (including self.res_nr_nodes_no_message).
        # Note that self.res_record only the records, when the number of nodes change.
        # Note that self.res_nr_nodes_no_message records with the time increasing.
        i = 1
        # time = 0 + para_time_granularity
        for i in range(1, self.true_time.size):
            is_found = False
            for j in range(len(self.res_record)):
                (t, nr_r, nr_i, nr_d) = self.res_record[j]
                if t > self.true_time[i]:
                    break
                # if t <= time
                # if [condition] the time in the record is less,
                # we should replicate the number of nodes into the list until the condition does not hold.
                # 对于record里面比当前时刻小的，可以不停的复制，直到停止
                else:
                    is_found = True
                    self.res_nr_nodes_no_message[i] = nr_r
                    self.res_nr_nodes_with_message[i] = nr_i
                    self.res_nr_nodes_selfish[i] = nr_d
            # generally, we should use the previous state in the temporal space,
            # in terms of the last time point.
            if not is_found:
                self.res_nr_nodes_no_message[i] = self.res_nr_nodes_no_message[i - 1]
                self.res_nr_nodes_with_message[i] = self.res_nr_nodes_with_message[i - 1]
                self.res_nr_nodes_selfish[i] = self.res_nr_nodes_selfish[i - 1]

    @staticmethod
    # Get the next contact time, i.e., the duration between two contacts,
    # which is drawn from Poisson distribution
    def get_next_wd(pl):
        res = np.random.exponential(1 / pl)
        return res

    def __init_contact_time(self):
        # Update the next contact time for every node in these N nodes.
        # these nodes will receive the message, after contact the assumed 'src' node.
        for i in range(self.N):
            # The time, at which node i contacts node 'src', is:
            # Note that at the initialization phase, which equals 0 + next_contact_time.
            self.src_nextContact[i] = self.running_time + self.get_next_wd(self.lam)
            # append every contact event to the Event List
            self.list_nextEvent.append((self.src_nextContact[i], event_fromsrc, i))

        # update the detection event, e.g., once detection, twice detection, ...
        # Note that only '1' contact is meaningful, '0.5' contact is meaningless.
        # Note that in tmp_list_U, the time is updated according to the increment of h
        tmp_list_U = []
        # self.U_star is the list of Integers, e.g., [0., 1., 0.5, 1, ....]
        assert self.h * len(self.U_star) == self.T
        # convert the U_star list into [(time, U_star), ... ]
        for i in range(len(self.U_star)):
            # a value of U_star will keep constant in [i*h, (i+1)*h]
            time = i*self.h
            tmp_list_U.append((time, self.U_star[i]))

        # Note that in timeindex, the time is updated
        # according to the increment of para_time_granularity (0.1)
        timeindex = 0.
        # get the list tmp_list_U_final, i.e., [(0, U_star), (0.1, U_star), ...]
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
        # Note that only 'once' detection is useful. '0.5' contact is meaningless.
        # get the Detection Event list.
        # i.e., [(0, U_star), (0.1, U_star), ...] into [(time, once_detect_event)]
        for (timeindex, value) in tmp_list_U_final:
            tmp = tmp + value*para_time_granularity
            if tmp >= 1.:
                self.list_nextEvent.append((timeindex, event_detect))
                tmp = tmp - 1

        # get the being selfish Event list with the similar logic
        tmp_list_rho = []
        assert self.h*len(self.rho_star) == self.T
        for i in range(len(self.rho_star)):
            # a value of rho_star will keep constant in [i*h, (i+1)*h]
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
                self.list_nextEvent.append((timeindex, event_selfish))
                tmp = tmp - 1

        # Sort All the Events, according to the temporal sequence.
        self.list_nextEvent.sort()

    # get the number of node in the 3 different states.
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
        # get the node list at the specific state 'state'
        index_list = np.argwhere(self.stateNode == state)
        index_list = np.squeeze(index_list, axis=1)
        index_list = index_list.tolist()
        # shuffle the node list, then we can choose a node from the node set in this state
        np.random.shuffle(index_list)
        return tmp_i, index_list

    # conduct the event.
    def run(self):
        if self.list_nextEvent[0][1] == event_detect:
            # print(self.list_nextEvent[0])
            # print(self.running_time, self.list_nextEvent)
            (t, eve) = self.list_nextEvent[0]
            assert self.running_time <= t
            # update running time; conduct the 'detection' event; pop this event
            self.running_time = t
            self.list_nextEvent.pop(0)
            target_detect = np.random.randint(0, self.N)
            if self.stateNode[target_detect] == state_selfish:
                self.stateNode[target_detect] = state_without_message
            self.res_record.append(self.update_nr_nodes_record_with_time())
        elif self.list_nextEvent[0][1] == event_fromsrc:
            # print(self.list_nextEvent[0])
            # print(self.running_time, self.list_nextEvent)
            (t, eve, i) = self.list_nextEvent[0]
            assert self.running_time <= t
            # update running time; conduct the 'contact' event; pop this event
            self.running_time = t
            self.list_nextEvent.pop(0)
            if self.stateNode[i] != state_with_message:
                self.stateNode[i] = state_with_message
            self.update_to_from_src(i)
            self.res_record.append(self.update_nr_nodes_record_with_time())
        elif self.list_nextEvent[0][1] == event_selfish:
            # print(self.list_nextEvent[0])
            # print(self.running_time, self.list_nextEvent)
            (t, eve) = self.list_nextEvent[0]
            assert self.running_time <= t
            # update running time; conduct the 'being selfish' event; pop this event
            self.running_time = t
            self.list_nextEvent.pop(0)
            target_detect = np.random.randint(0, self.N)
            if self.stateNode[target_detect] == state_with_message:
                self.stateNode[target_detect] = state_selfish
            self.res_record.append(self.update_nr_nodes_record_with_time())
            # self.res_record.append(self.update_nr_nodes_record_with_time())
        else:
            print('Internal Err! -- unkown event time:{} eve_list:{}'.format(self.running_time, self.list_nextEvent))

    def update_to_from_src(self, i):
        tmp_next_time = self.get_next_wd(self.lam) + self.running_time
        self.src_nextContact[i] = tmp_next_time
        loc = 0
        for loc in range(len(self.list_nextEvent)):
            if self.list_nextEvent[loc][0] >= tmp_next_time:
                break
        self.list_nextEvent.insert(loc, (tmp_next_time, event_fromsrc, i))

    # return the time index, the true time, the number of nodes in '3' states at different time.
    def get_sim_res(self):
        return self.time_index, self.true_time, self.res_nr_nodes_no_message, \
               self.res_nr_nodes_with_message, self.res_nr_nodes_selfish

# get the payoff based on the simulation result
def get_sim_payoff(N, T, r, i, d, U, rho, h):
    K0 = np.sum(d) * gamma * para_time_granularity
    K1 = np.sum(U) * alpha * h
    K2 = np.sum(rho) * beta * h
    res = K0 + K1 - K2
    return res

def strlistTofloatlist(in_ll):
    res_ll = []
    for term in in_ll:
        tmp_res = float(term)
        res_ll.append(tmp_res)
    return res_ll


# Section 1: read analytical result
filename = 'RecursiveMth_AnalyRes_div{}.csv'.format(div)
f = open(filename, 'r')
ll = f.readline()
time_list_ex = strlistTofloatlist(ll.split(','))
ll = f.readline()
I_list_ex = strlistTofloatlist(ll.split(','))
ll = f.readline()
D_list_ex = strlistTofloatlist(ll.split(','))
ll = f.readline()
I_upper_bound_list = strlistTofloatlist(ll.split(','))
ll = f.readline()
I_lower_bound_list = strlistTofloatlist(ll.split(','))
ll = f.readline()
D_upper_bound_list = strlistTofloatlist(ll.split(','))
ll = f.readline()
D_lower_bound_list = strlistTofloatlist(ll.split(','))
ll = f.readline()
U_list = strlistTofloatlist(ll.split(','))
ll = f.readline()
rho_list = strlistTofloatlist(ll.split(','))
f.close()
assert len(U_list) == div
assert len(rho_list) == div
# analytical cost
ana_cost = 0.
for i in range(div):
    ana_cost = ana_cost + ((D_list_ex[i] + D_list_ex[i+1])/2*gamma + alpha * U_list[i] - beta * rho_list[i])*h
print('analytical cost:{}'.format(ana_cost))

# Section 2: calculate simulation result
per = int(1 / para_time_granularity)
multi_times_r = np.zeros((run_times, T*per))
multi_times_i = np.zeros((run_times, T*per))
multi_times_d = np.zeros((run_times, T*per))
# payoff
multi_times_payoff = np.zeros(run_times)
time_idx = []
time_true = []
for k in range(run_times):
    print('running time:{}'.format(k))
    # conduct the simulations
    the = Sim(N, T, lam, h, U_list, rho_list)
    time_idx, time_true, r, i, d = the.get_sim_res()
    payoff = get_sim_payoff(the.N, T, r, i, d, U_list, rho_list, h)
    # print('payoff:{}'.format(payoff))
    multi_times_payoff[k] = payoff
    # print('cost_from_sel:{} cost_from_detect:{}'.format(cost1, cost2))
    multi_times_r[k, :] = r
    multi_times_i[k, :] = i
    multi_times_d[k, :] = d
Sim_R_list = np.sum(multi_times_r, axis=0) / run_times
Sim_I_list = np.sum(multi_times_i, axis=0) / run_times
Sim_D_list = np.sum(multi_times_d, axis=0) / run_times
Sim_Cost = np.sum(multi_times_payoff, axis=0) / run_times
print('mean payoff:{}'.format(Sim_Cost))
# payoff = get_sim_payoff(para_N,r,i,d,U_star,rho_star)
# print('avg payoff:{}'.format(payoff))
# print('\n')
# return r, i, d

# Section 3: store the results (analytical and sim) to *.csv
def ConvertoString(ll):
    output_str = ''
    for i in range(len(ll)-1):
        output_str = output_str + str(ll[i]) + ','
    output_str = output_str + str(ll[len(ll)-1]) + '\n'
    return output_str

filename = 'RecursiveMth_AllSimRes_div{}.csv'.format(div)
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

f.write(ConvertoString(time_true))
f.write(ConvertoString(Sim_I_list))
f.write(ConvertoString(Sim_D_list))
f.write(ConvertoString(Sim_R_list))
f.close()

if isDraw:
    plt.figure('Sim IDR')
    # plot mean value of simulated result
    # x_true = np.arange(0, T * per) * para_time_granularity
    plt.plot(time_true, Sim_I_list, 'b', label='I')
    plt.plot(time_true, Sim_D_list, 'r', label='D')
    plt.plot(time_true, Sim_R_list, 'g', label='R')

    # plot analytical result
    I_test = np.array(I_list_ex)
    D_test = np.array(D_list_ex)
    R_test = N - I_test - D_test
    plt.plot(time_list_ex, I_test, 'b--o', label='I analysis')
    plt.plot(time_list_ex, D_test, 'r--s', label='D analysis')
    plt.plot(time_list_ex, R_test, 'g--*', label='R analysis')

    plt.plot(time_list_ex, I_lower_bound_list, 'b-v', label='I lower')
    plt.plot(time_list_ex, I_upper_bound_list, 'b-^', label='I upper')

    plt.plot(time_list_ex, D_lower_bound_list, 'r-v', label='D lower')
    plt.plot(time_list_ex, D_upper_bound_list, 'r-^', label='D upper')

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Expected number of nodes")
    # plt.show()

    # plot U rho control
    plt.figure('U rho')
    x = np.zeros((div*2, 3), dtype='float')
    for i in range(div):
        x[i*2, 0] = i*h
        x[i*2+1, 0] = i*h+0.99*h
        x[i*2, 1] = U_list[i]
        x[i*2+1, 1] = U_list[i]
        x[i*2, 2] = rho_list[i]
        x[i*2+1, 2] = rho_list[i]
    plt.plot(x[:,0], x[:,1], 'b-o',label=r'U(t)')
    plt.plot(x[:,0], x[:,2], 'r-^', label=r'$\rho(t)$')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Behaviors")
    # plt.show()

    # plot analytical I and D
    plt.figure('IDR bk fw')
    I_test = np.array(I_test)
    D_test = np.array(D_test)
    R_test = N - I_test - D_test
    plt.plot(time_list_ex, I_test, 'b--*', label='I fw')
    plt.plot(time_list_ex, D_test, 'r--*', label='D fw')
    plt.plot(time_list_ex, R_test, 'g--*', label='R fw')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

