# preprocessing data
import pandas as pd
import numpy as np

raw_data = pd.read_csv("./data/train.csv", encoding='big5')
all_data = raw_data.iloc[:, 3:]
# 将NR替换为0
all_data[all_data == 'NR'] = 0
data = all_data.to_numpy()

# 将原始的资料重组为12个连续的数据组
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample


# 每个月480个小时，每9小时形成一个data，每个月会有471个data，所以一共有 471 * 12 个data
x = np.empty([471 * 12, 18 * 9], dtype=float)
y = np.empty([471 * 12, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]

# print(x)
# print(y)


# Normalization
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(471 * 12):
    for j in range(18 * 9):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
# End Normalization

x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
# Split train set into train set and validation set
import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
x_valid_set = x[math.floor(len(x) * 0.8):, :]
y_train_set = y[: math.floor(len(x) * 0.8), :]
y_valid_set = y[math.floor(len(x) * 0.8):, :]
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_valid_set))
# print(len(y_valid_set))


# training
dim = 18 * 9 + 1    # add b
w = np.zeros([dim, 1])
learning_rate = 10
iter_time = 5000
adagrad = np.zeros([dim, 1])
eps = 0.0000001
lmd = 100
for t in range(iter_time):
    loss = (np.sum(np.power(w, 2)) + np.sum(np.power(y_train_set - np.dot(x_train_set, w), 2))) / (471 * 12)
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set) + 2 * lmd * w
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    if t % 100 == 0:
        print(str(t) + ": loss = " + str(loss))

# validation
loss = (np.sum(np.power(w, 2)) + np.sum(np.power(y_valid_set - np.dot(x_valid_set, w), 2))) / (471 * 12)
print("\nvalid loss = " + str(loss))


# preprocessing testdata
test_raw_data = pd.read_csv("./data/test.csv", header=None, encoding='big5')
test_data = test_raw_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(240):
    for j in range(18 * 9):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

ans_y = np.dot(test_x, w)
ans_y = np.around(ans_y, decimals=1)

import csv
with open("./answer.csv", mode='w', newline='') as f:
    f_writer = csv.writer(f)
    header = ['id', 'value']
    print(header)
    f_writer.writerow(header)
    for i in range(240):
        row = ["id_" + str(i), str(ans_y[i][0])]
        f_writer.writerow(row)
        print(row)
f.close()
