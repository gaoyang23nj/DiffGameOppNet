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
import datetime
import matplotlib.pyplot as plt

# # the number of nodes
# N = 100
# # the frequency of node concacts
# lam = 0.004
#
# # --20241201 infocom'05_dataset # id:1 num_interval:644 avg_value:383.01863354037266 estimated_Lambda:0.0026108390360976876
# N = 40
# lam = 0.0026108390360976876
#
# # --20241203 RWP_dataset # id:0 num_interval:2754 avg_value:313.4738562091503 estimated_Lambda:0.003190058692909938
# N = 100
# lam = 0.003
# # --20241204 RWP_dataset id:20 num_interval:112 avg_value:629.7410714285714 estimated_Lambda:0.001587954232890502
N = 100
lam = 0.001702483720762598

gamma = 1
alpha = 1
beta = 0.1

div = 10
T = 300
h = T/div

run_times = 200
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
    def __init__(self, num_nodes, total_time, lam, h, U_star, rho_star, begin_time, src_id):
        # the given detection strategy, which is a list
        self.U_star = U_star
        # the given selfish strategy, which is a list
        self.rho_star = rho_star

        # parameters
        self.lam = lam
        self.T = total_time
        self.N = num_nodes
        self.h = h

        self.src_id = src_id
        # the current time in the simulation
        # self.running_time = 0
        self.begin_time = begin_time
        self.running_time = begin_time
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
            if self.list_nextEvent[0][0] >= self.T+self.begin_time:
                break
            # conduct the 3 events (contact, being selfish, detection); update the running time;
            self.run()


        # arrange true time to timeindex, 真实时间从 [begin_time, T+begin_time], 统计结果处理[0,T] step 0.1
        # 真实时间统一减去begin_time
        tmp_res_record = []
        for item in self.res_record:
            (t, nr_r, nr_i, nr_d) = item
            tmp_res_record.append((t-begin_time, nr_r, nr_i, nr_d))

        # Convert the Results from self.res_record to 3 matrix (including self.res_nr_nodes_no_message).
        # Note that self.res_record only the records, when the number of nodes change.
        # Note that self.res_nr_nodes_no_message records with the time increasing.
        i = 1
        # time = 0 + para_time_granularity
        for i in range(1, self.true_time.size):
            is_found = False
            for j in range(len(tmp_res_record)):
                (t, nr_r, nr_i, nr_d) = tmp_res_record[j]
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

    def process_Helsinkcontact(self, src_id):
        # infocom'05 data id:1~41
        assert((src_id >= 0)and(src_id<100))
        count = 0
        with open('.\Helsink_dataset\encohist_20241204094635.enc') as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                line = line.strip()
                # parse
                strs = line.split(',')
                s = int(strs[0])
                d = int(strs[1])
                time = int(strs[2])
                # print(strs)
                # filter illegal records
                if ((s != src_id) or (d >= 100) or (d < 0)):
                    continue
                # id exchange
                if d < src_id:
                    pass
                elif d > src_id:
                    d = d - 1
                else:
                    print('Interal Err！id err in process_pair({}) line:{}'.format(src_id, line))
                # RWP time sim_TimeStep:0.1
                self.list_nextEvent.append((time*0.1, event_fromsrc, d))
                count = count + 1
        print('total count:{}'.format(count))

    def __init_contact_time(self):
        # Update the next contact time for every node in these N nodes.
        # these nodes will receive the message, after contact the assumed 'src' node.
        # for i in range(self.N):
        #     # The time, at which node i contacts node 'src', is:
        #     # Note that at the initialization phase, which equals 0 + next_contact_time.
        #     self.src_nextContact[i] = self.running_time + self.get_next_wd(self.lam)
        #     # append every contact event to the Event List
        #     self.list_nextEvent.append((self.src_nextContact[i], event_fromsrc, i))
        # self.process_infocom05contact(self.src_id)
        self.process_Helsinkcontact(self.src_id)

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
                self.list_nextEvent.append((self.begin_time + timeindex, event_detect))
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
                self.list_nextEvent.append((self.begin_time + timeindex, event_selfish))
                tmp = tmp - 1

        # Sort All the Events, according to the temporal sequence.
        self.list_nextEvent.sort()
        # clean the old events in the dataset
        event_list_bk = self.list_nextEvent.copy()
        del_id = []
        for item in range(len(event_list_bk)):
            # if ((event_list_bk[item][0] < self.begin_time) or (event_list_bk[item][0] > self.T + self.begin_time)):
            if event_list_bk[item][0] < self.begin_time:
                del_id.append(item)
        print('preprocess delete {} records'.format(len(del_id)))
        del_id.reverse()
        for item in del_id:
            self.list_nextEvent.pop(item)

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
            # 0~N-1
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
            # self.update_to_from_src(i)
            self.res_record.append(self.update_nr_nodes_record_with_time())
        elif self.list_nextEvent[0][1] == event_selfish:
            # print(self.list_nextEvent[0])
            # print(self.running_time, self.list_nextEvent)
            (t, eve) = self.list_nextEvent[0]
            assert self.running_time <= t
            # update running time; conduct the 'being selfish' event; pop this event
            self.running_time = t
            self.list_nextEvent.pop(0)
            # 0~N-1
            target_detect = np.random.randint(0, self.N)
            if self.stateNode[target_detect] == state_with_message:
                self.stateNode[target_detect] = state_selfish
            self.res_record.append(self.update_nr_nodes_record_with_time())
            # self.res_record.append(self.update_nr_nodes_record_with_time())
        else:
            print('Internal Err! -- unkown event time:{} eve_list:{}'.format(self.running_time, self.list_nextEvent))

    # def update_to_from_src(self, i):
    #     tmp_next_time = self.get_next_wd(self.lam) + self.running_time
    #     self.src_nextContact[i] = tmp_next_time
    #     loc = 0
    #     for loc in range(len(self.list_nextEvent)):
    #         if self.list_nextEvent[loc][0] >= tmp_next_time:
    #             break
    #     self.list_nextEvent.insert(loc, (tmp_next_time, event_fromsrc, i))

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

# Section 3: store the results (analytical and sim) to *.csv
def ConvertoString(ll):
    output_str = ''
    for i in range(len(ll)-1):
        output_str = output_str + str(ll[i]) + ','
    output_str = output_str + str(ll[len(ll)-1]) + '\n'
    return output_str

def main_simulate(U_list, rho_list):
    assert len(U_list) == div
    assert len(rho_list) == div
    # # calculating analytical cost
    # ana_cost = 0.
    # for i in range(div):
    #     ana_cost = ana_cost + ((D_list_ex[i] + D_list_ex[i+1])/2*gamma + alpha * U_list[i] - beta * rho_list[i])*h
    # print('analytical cost:{}'.format(ana_cost))
    # Section 2: calculate simulation result
    per = int(1 / para_time_granularity)
    multi_times_r = np.zeros((run_times, T*per))
    multi_times_i = np.zeros((run_times, T*per))
    multi_times_d = np.zeros((run_times, T*per))
    # payoff
    multi_times_payoff = np.zeros(run_times)
    time_idx = []
    time_true = []
    begin_times = 0. + np.random.random(run_times) * 1000
    # begin_times = [0.]*run_times
    for k in range(run_times):
        print('running time:{}'.format(k))
        # conduct the simulations
        # src_id = np.random.randint(1, 42)
        src_id = np.random.randint(0, 100)
        the = Sim(N, T, lam, h, U_list, rho_list, int(begin_times[k]),src_id)
        time_idx, time_true, r, i, d = the.get_sim_res()
        payoff = get_sim_payoff(the.N, T, r, i, d, U_list, rho_list, h)
        print('payoff:{}'.format(payoff))
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
    return Sim_Cost



if __name__ == "__main__":
    List_Res = []
    # 1.[NE strategy] U_list, rho_list are calcualted by obtain_Analytical_J.py
    U_list = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    rho_list = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    t1 = datetime.datetime.now()
    print('begin at:{}'.format(t1))
    tmp_res = main_simulate(U_list, rho_list)
    List_Res.append(tmp_res)
    t2 = datetime.datetime.now()
    print('end at:{}; Delta:{}'.format(t2, t2-t1))

    # # 2.[Umin (off) strategy] best_reply
    U_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rho_list = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    t1 = datetime.datetime.now()
    print('begin at:{}'.format(t1))
    tmp_res = main_simulate(U_list, rho_list)
    List_Res.append(tmp_res)
    t2 = datetime.datetime.now()
    print('end at:{}; Delta:{}'.format(t2, t2-t1))

    # # 3.[Umax (on) strategy] best_reply
    U_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    rho_list = [0., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    t1 = datetime.datetime.now()
    print('begin at:{}'.format(t1))
    tmp_res = main_simulate(U_list, rho_list)
    List_Res.append(tmp_res)
    t2 = datetime.datetime.now()
    print('end at:{}; Delta:{}'.format(t2, t2-t1))

    # 4.[half Umax strategy] best_reply
    U_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    rho_list = [0., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    t1 = datetime.datetime.now()
    print('begin at:{}'.format(t1))
    tmp_res = main_simulate(U_list, rho_list)
    List_Res.append(tmp_res)
    t2 = datetime.datetime.now()
    print('end at:{}; Delta:{}'.format(t2, t2-t1))

    # # 5.[half Umax strategy] best_reply
    U_list = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    rho_list = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    t1 = datetime.datetime.now()
    print('begin at:{}'.format(t1))
    tmp_res = main_simulate(U_list, rho_list)
    List_Res.append(tmp_res)
    t2 = datetime.datetime.now()
    print('end at:{}; Delta:{}'.format(t2, t2-t1))

    print('#'*40)
    for item in List_Res:
        print(item)
