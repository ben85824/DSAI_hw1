import numpy as np
import sys
from keras.utils import to_categorical
from datetime import datetime

# train data: 2016 - 2017
# val data: 2018
def print_schedule(string):
    sys.stdout.flush()
    sys.stdout.write(string + '\r')

def get_week_day(string):
    date = datetime.strptime(string,'%Y%M%d')
    return date.weekday()

def load_csv(name, index):
    with open(name, 'r', encoding = 'utf-8-sig') as f:
        data = []
        for row in f:
            row = row.split(',')
            row = [row[x] for x in index]
            data.append(row)
        return data

def get_train_data(data, pred_days, train_days):
    x, y, x2 = [],[],[]
    total = len(data) - pred_days - train_days
    for i in range(len(data) - pred_days - train_days):
        print_schedule('%.3f'%(i/total))
        input_data, pred = [], []
        pred_week_days = []
        for j in range(train_days):
            days_info = [float(data[j + i][x]) for x in [1, 2, 3]]
            days_info[0]/= 10000
            days_info[1]/= 100
            days_info[2]/= 100
            week_day = to_categorical(get_week_day(data[j + i][0]), 7)
            days_info.extend(week_day)
            input_data.append(days_info)
        for j in range(pred_days):
            days_info = float(data[j + i + train_days][1])
            days_info /= 10000
            week_day = get_week_day(data[j + i + train_days][0])
            pred.append(days_info)
            pred_week_days.append(to_categorical(week_day, 7))
        x.append(input_data)
        y.append(pred)
        x2.append(pred_week_days)
    return np.array(x), np.array(y), np.array(x2)
        

train_days = 14
pred_days = 7
train_col_idx = [0, 2, 5, 6]

train_data = load_csv('2016-2017data.csv', train_col_idx)
test_data = load_csv('2018data.csv', train_col_idx)[1:]
train_x, train_y, train_x2 =  get_train_data(train_data, pred_days, train_days)
test_x, test_y, test_x2 = get_train_data(test_data, pred_days, train_days)
np.save('train_x.npy', train_x)
np.save('train_y.npy', train_y)
np.save('train_x2.npy', train_x2)
np.save('test_x.npy', test_x)
np.save('test_y.npy', test_y)
np.save('test_x2.npy', test_x2)



    





        

