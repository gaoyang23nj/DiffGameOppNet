# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt




def process_pair():
    count = 0
    with open('.\RWP_dataset\encohist_20241203064557.enc') as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            line = line.strip()
            # parse
            strs = line.split(',')
            s = int(strs[0])
            d = int(strs[1])
            t = int(strs[2])
            # print(strs)
            idx = s*N+d
            # print(dict_contact_pair[idx], s, d)
            assert (dict_contact_pair[idx]==(s,d))
            # print('{} {} {}'.format(s,d,t))
            list_contact_pair[idx].append(t)
            count = count + 1
    print('total count:{}'.format(count))

def cal_for_srcid(id, isDraw):
    from_idx = id * N
    to_idx = (id+1) * N
    print('cal for id_{} from:{} to:{}'.format(id,from_idx,to_idx))
    tmp_list = []
    tmp_count = 0
    for curr_idx in range(from_idx, to_idx):
        tmp_list.extend(list_contact_pair[curr_idx].copy())
        print('idx:{} num_contacts:{}'.format(curr_idx,len(list_contact_pair[curr_idx])))
        tmp_count = tmp_count + len(list_contact_pair[curr_idx])
    tmp_list.sort()
    print('num_count:{} begin at:{} end at:{}'.format(tmp_count, tmp_list[0], tmp_list[-1]))
    # plot pdf distribution
    cdf = []
    for next in range(1, len(tmp_list)):
        interval = tmp_list[next] - tmp_list[next - 1]
        assert (interval >= 0)
        cdf.append(interval)
    cdf.sort()
    # cdf[len(cdf)-1]/100
    step = 10
    _tmp = 0
    x = []
    y = []
    while (True):
        x.append(_tmp)
        _value = np.sum(np.array(cdf) <= _tmp) / len(cdf)
        y.append(_value)
        if _value > 0.95:
            break
        _tmp = _tmp + step
    e_lambda = 1/np.average(cdf)
    print('id:{} num_interval:{} avg_value:{} estimated_Lambda:{}'.format(id, len(cdf), np.average(cdf), e_lambda))

    if isDraw:
        plt.figure(1)
        plt.plot(x, y)
        plt.show()

    return e_lambda

N = 100
dict_contact_pair = []
list_contact_pair = []
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i in range(N):
        # print(i+1)
        for j in range(N):
            # print(j + 1)
            dict_contact_pair.append((i,j))
            list_contact_pair.append([])
    print(len(dict_contact_pair))
    process_pair()

    # calculate interval time, estimate \lambda for node_id
    id = 1
    # cal_for_srcid(id)
    lambda_list = []
    for id in range(100):
        e_lambda = cal_for_srcid(id, False)
        lambda_list.append(e_lambda)
    print(lambda_list)
    print(np.array(lambda_list).mean())
    print('ok')
