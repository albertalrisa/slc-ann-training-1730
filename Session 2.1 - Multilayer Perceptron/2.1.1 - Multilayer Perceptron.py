import tensorflow as tf
import numpy as np
import csv
from datetime import datetime

# 1. Prepare Dataset
def load_dataset(filepath):
    dataset = []
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            feature = row[1:5]
            label = row[5]
            dataset.append((feature, label))
    return dataset

label_list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
def preprocess_data(dataset):
    preprocessed = []
    for feature, label in dataset:
        # Convert each item to float
        feature = np.array([float(x) for x in feature])

        label_index = label_list.index(label)
        label = np.zeros(len(label_list), 'int')
        label[label_index] = 1

        preprocessed.append((feature, label))
    return preprocessed

dataset = load_dataset('Iris.csv')
dataset = preprocess_data(dataset)

# 2. Create architecture
number_of_input = 4
number_of_hidden = [4]
number_of_output = 3

def fully_connected(input, nodes_in, nodes_out, name="fc"):
    with tf.name_scope(name):
        W = tf.Variable(tf.random_normal([nodes_in, nodes_out]), name="W")
        b = tf.Variable(tf.random_normal([nodes_out]), name="b")
        Wx_b = tf.matmul(input, W) + b
        act = tf.nn.sigmoid(Wx_b)
        tf.summary.histogram('weight', W)
        tf.summary.histogram('bias', b)
        tf.summary.histogram('sigmoid', act)
        return act

def build_model(input):
    layer_1 = fully_connected(input, number_of_input, number_of_hidden[0], "layer1")
    layer_o = fully_connected(layer_1, number_of_hidden[0], number_of_output, "layer2")
    return layer_o

# [None, number_of_input] => 
x = tf.placeholder(tf.float32, [None, number_of_input], name="input")
t = tf.placeholder(tf.float32, [None, number_of_output], name="target")
lr = .01
number_of_epoch = 10000
report_between = int(.1 * number_of_epoch)

def optimize(model, dataset):
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(.5 * (t - model) ** 2)
        tf.summary.scalar('loss', cost)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.arg_max(model, 1), tf.arg_max(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = './log/' + time
    
    # TensorBoard -> writer
    writer = tf.summary.FileWriter(log_dir)

    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for i in range(number_of_epoch):
            features = [data[0] for data in dataset]
            labels = [data[1] for data in dataset]

            feed_dict = {x: features, t: labels}
            _, cost_val, acc_val = \
                sess.run([optimizer, cost, accuracy], feed_dict)

            if i % report_between == 0:
                print('[Epoch {:5}] Loss: {:1.8f} Acc: {:3.2f}%'
                .format(i, cost_val, acc_val * 100.0))

            if i % 5 == 0:
                [summary] = sess.run([merged_summary], feed_dict)
                writer.add_summary(summary, i)

model = build_model(x)
optimize(model, dataset)
