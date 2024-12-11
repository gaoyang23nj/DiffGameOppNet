# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt


def process_pair():
    count = 0
    with open('.\infocom05_dataset\contacts.Exp3.dat') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # parse
            strs = line.split('\t')
            s = int(strs[0])
            d = int(strs[1])
            t = int(strs[2])
            # print(strs)
            if s>41 or d>41:
                continue
            idx = (s-1)*41+d-1
            # print(dict_contact_pair[idx], s, d)
            assert (dict_contact_pair[idx]==(s,d))
            # print('{} {} {}'.format(s,d,t))
            list_contact_pair[idx].append(t)
            count = count + 1
    print('total count:{}'.format(count))

def cal_for_srcid(id, isDraw):
    from_idx = (id - 1) * 41
    to_idx = id * 41
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

dict_contact_pair = []
list_contact_pair = []
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i in range(41):
        # print(i+1)
        for j in range(41):
            # print(j + 1)
            dict_contact_pair.append((i+1,j+1))
            list_contact_pair.append([])
    print(len(dict_contact_pair))
    process_pair()

    # calculate interval time, estimate \lambda for node_id
    id = 1
    lambda_list = []
    # cal_for_srcid(id)
    for id in range(1, 42):
        e_lambda = cal_for_srcid(id, False)
        lambda_list.append(e_lambda)
    print(lambda_list)
    print(np.array(lambda_list).mean())

    print('ok')
