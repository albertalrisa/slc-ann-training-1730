import numpy as np
from random import choice

dataset = [
    (np.array([1, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([0, 1]), np.array([1])),
    (np.array([0, 0]), np.array([0])),
]

number_of_input = 2
number_of_output = 1
number_of_iter = 200
report_between = int(.1 * number_of_iter)
lr = .5

weights = {
    'w': np.random.normal(size=[number_of_input, number_of_output])
}

biases = {
    'b1': np.random.normal(size=[number_of_output])
}

def hard_limit(u, threshold):
    u[u >= threshold] = 1
    u[u < threshold] = 0
    return u

def forward_pass(x):
    u = np.matmul(x, weights['w']) + biases['b1']
    return u

def deriveMSE(target, y):
    return -(target - y)

def deriveWxToW(x):
    return x

def deriveErrorToWeight(target, y, x):
    return deriveMSE(target, y) * deriveWxToW(x)

def deriveErrorToBias(target, y):
    return deriveMSE(target, y)

def optimize(dataset):
    for i in range(number_of_iter):
        x, target = choice(dataset)
        y = hard_limit(forward_pass(x), .5)
        error = .5 * (target - y) ** 2

        weights['w'] -= lr * deriveErrorToWeight(target, y, x.reshape([2, 1]))
        biases['b1'] -= lr * deriveErrorToBias(target, y)

        if i % report_between == 0:
            correct_prediction = 0
            for data in dataset:
                if data[1] == hard_limit(forward_pass(data[0]), .5):
                    correct_prediction += 1
                accuracy = correct_prediction / len(dataset) * 100
            print("[Epoch {:5}] Accuracy: {:3}%".format(i, accuracy))

optimize(dataset)

while True:
    user_input = input('Input numbers [0 | 1 | exit]: ')
    if(user_input.lower() == 'exit'):
        break
    user_input = user_input.split(' ')
    user_input = [int(x) for x in user_input]
    prediction = hard_limit(forward_pass(np.array(user_input)), .5)
    print("{} OR {} = {}".format(user_input[0], user_input[1], prediction))