import pandas as pd
import numpy as np
import csv

train_set_path = "./train_data.csv"
test_set_path = "./test_data.csv"


def read_train_data(file_path):
    raw_data = pd.read_csv(file_path).to_numpy().astype(float)
    all_data = raw_data[:raw_data.shape[0] - 1, :]

    x = all_data[:, :all_data.shape[1] - 2]
    y = all_data[:, all_data.shape[1] - 2:]

    return x, y


def normalize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    for j in range(x.shape[1]):
        x[:, j] = (x[:, j] - x_mean[j]) / (x_std[j] + 1e-8)

    return x


def softmax(z):
    z = np.exp(z)
    z_sum = np.sum(z, axis=1)
    for i in range(z_sum.shape[0]):
        z[i, :] = z[i, :] / z_sum[i]
    return z


def training(set_x, set_y, learning_rate=0.2, epoch=20):
    train_x = set_x[:round(set_x.shape[0] * 0.8), :]
    train_y = set_y[:round(set_y.shape[0] * 0.8), :]
    valid_x = set_x[round(set_x.shape[0] * 0.8):, :]
    valid_y = set_y[round(set_y.shape[0] * 0.8):, :]

    w = np.zeros((train_x.shape[1], train_y.shape[1]))
    b = np.zeros((1, train_y.shape[1]))

    w_grad_sum = np.full((w.shape[0], w.shape[1]), 1e-8)
    b_grad_sum = np.full((b.shape[0], b.shape[1]), 1e-8)

    # train
    for e in range(epoch):
        z = np.dot(train_x, w) + b
        y = softmax(z)
        error = train_y - y

        w_gradient = - np.dot(train_x.T, error)
        b_gradient = - np.sum(error, axis=0)
        w_grad_sum = w_grad_sum + w_gradient ** 2
        b_grad_sum = b_grad_sum + b_gradient ** 2

        w = w - learning_rate * w_gradient / np.sqrt(w_grad_sum)
        b = b - learning_rate * b_gradient / np.sqrt(b_grad_sum)

        # valid
        z = np.dot(valid_x, w) + b
        y = softmax(z)
        y_predict = np.round(y)
        loss = - np.sum(np.dot(valid_y.T, np.log(y)) / valid_y.shape[0], axis=0)

        # print(valid_y.shape, y.shape, loss.shape)

        print("After Epoch " + str(e))
        for i in range(len(loss)):
            print("Loss of Class " + str(i) + ": " + str(loss[i]))
        print("\n")

    return w, b


def predict(test_file_path, weight, bias):
    x = pd.read_csv(test_file_path).to_numpy().astype(float)
    x = normalize(x)
    z = np.dot(x, weight) + bias
    return np.around(softmax(z)).astype(int)


def write_answer(answer_file_path, answer):
    with open(answer_file_path, mode='w', newline='') as f:
        f_writer = csv.writer(f)
        header = ['id', 'label']
        print(header)
        f_writer.writerow(header)
        for i in range(answer.shape[0]):
            row = [str(i), str(answer[i][0])]
            f_writer.writerow(row)
            print(row)
    f.close()


if __name__ == "__main__":
    train_set_x, train_set_y = read_train_data(train_set_path)
    train_set_x = normalize(train_set_x)

    w, b = training(train_set_x, train_set_y, 0.00001, 15)
    y = predict(test_set_path, w, b)

    write_answer("./answer.csv", y)
